# StraTA（个人版 / 小模型方案）on Qwable-9B-Claude-Fable-5

Strategic Trajectory Abstraction（StraTA）分层智能体强化学习的**单卡个人复现版**：在
`empero-ai/Qwable-9B-Claude-Fable-5`（Qwen3.5-9B 全参微调，冻结视觉的多模态架构 + 混合
Gated DeltaNet 线性注意力 + `<think>` 推理）上，做编码智能体的 SFT 格式对齐 → GRPO 强化学习。

本仓库记录了从"环境搭建失败"到"端到端跑通"的完整过程，**重点是定位并修复了一系列阻塞性
bug**（见下文「关键修复」）。

---

## 1. 方法概览（StraTA）

分层 agentic RL，一个训练 step 分五阶段：

1. **策略采样**：对任务用 `STRATEGY_PROMPT` 采样 `sigma*N` 个候选策略，FPS 多样性选择取 `N` 个。
2. **分层 Rollout**：每条策略跑 `M` 个 episode（在 `CodeGym` 沙盒里多步交互）。
3. **奖励计算**：策略级用 top-δ 聚合；动作级用 **跨 rollout 的 group-relative 优势（正确 GRPO）** +
   长度惩罚 + 格式惩罚 + 自我评判（self-judge）惩罚 κ。
4. **旧策略 log-prob 采集**（no_grad，批量化）。
5. **GRPO clipped-surrogate loss + KL 近似**（无参考模型，省 ~5GB 显存），增量/批量反向。

---

## 2. 仓库结构

```
src/
  strata_trainer.py   # 核心：StraTAConfig / StraTATrainer / 提示模板 / 奖励 / 批量化生成与训练
  sandbox.py          # CodeGym：subprocess 沙盒（write/read/bash/test 动作 DSL，终端稀疏奖励）
  sft_train.py        # 协议匹配的 SFT 格式对齐（strategy/action 两种样本，assistant-only loss）
  eval.py             # 评估：成功率 + 动作格式有效率（自动注入验证协议）
  eval_hint.py        # 带显式验证协议注入的评估（用于对照/调试）
  prepare_data.py     # 数据生成：合成任务 + HumanEval + MBPP + SFT 数据
  smoke_test.py       # 端到端冒烟（小 N/M/步数）
  quick_test.py       # 模型加载 + 显存自检
  probe_batch.py      # 探针：验证批量/left-pad 生成在自定义架构上可用 + 测速
configs/
  full.json           # 生产 RL 配置（warm-start SFT，batch_size=2，max_steps=100）
  smoke*.json         # 各阶段冒烟配置
data/
  train/              # all_tasks.json(364) synthetic.json humaneval.json sft_data.json(300)
  eval/               # eval_tasks.json(100) eval_small.json(12)
```

> 模型权重（`model/`，18GB）与训练产物（`checkpoints/`）不入库，按下方「运行」步骤自行下载/生成。

---

## 3. 关键修复（从"跑不通"到"端到端跑通"）

复现最初完全跑不起来，根因与修复如下（按发现顺序）：

### 3.1 chat-template 训练/推理不一致 → action 全空、0% 解出
- `generate()` 喂的是**原始文本**，而模型（Qwen3.5 系）训练分布是 chat 模板 → 输入 OOD，生成乱码/空。
- **修复**：`generate()` / log-prob / loss 全部先 `apply_chat_template(..., add_generation_prompt=True)`
  再 tokenize；SFT 数据也按 per-prompt 协议重建（不再把多轮拼成一条）。

### 3.2 `_collect_log_probs` 返回错误变量 → 训练崩溃
- 返回了未定义的 `log_probs` 而非累加列表。**修复**：返回 `log_probs_list`。

### 3.3 空目标 / 非有限 loss → NaN 腐蚀 LoRA 权重
- 空目标做 `.mean()` 产生 NaN 并反向。**修复**：空目标过滤 + `torch.isfinite` 守卫，非有限直接跳过。

### 3.4 ★验证协议错配（决定性的解题率修复）
- 沙盒 `_check_success` 跑 `task["test_command"]` = `python3 -c "from solution import ..."`，
  因此模型**必须**写到 `solution.py`；但提示从没告诉它（SFT 数据是通用的，写 `main.py`）。
  模型把**正确的代码写进了错误的文件** → import 失败 → 永远 solved=False。
- **修复**：`augment_description(task)` 把验证命令 + "写入被 import 的模块（如 solution.py）"
  + 动作 DSL 追加到任务描述，在 `training_step` / `_run_episode` / `evaluate` / `eval.py`
  **三处一致**注入（避免再造训练/推理偏移）。
- **效果**（同一 SFT 模型，零额外训练）：eval_small 解题率 **12.5% → 62.5%**，动作格式从 0% 空变为真实代码。

### 3.5 ★GRPO 信用分配写错 → 训练无信号（Items=0）
- 原代码对**单个 rollout 内的时间步**算 group-relative 优势；而绝大多数任务第 1 步就解出
  （`len(rewards)==1`）→ `compute_advantages` 返回全 0 → 所有 item 被零优势过滤 → 不训练。
- **修复**：正确的 GRPO——优势在**同一策略的 M 个 rollout 之间**计算（group = M rollouts），
  每个 rollout 的所有动作继承该 rollout 的组相对优势。配合 `batch_size>1` 保证每步大概率有非零信号。

### 3.6 生成/rollout/log-prob 全串行 → 显存闲置、极慢
- 48GB 卡只用了 ~19GB，串行生成把卡浪费了 ~8-12×。
- **修复**（探针 `probe_batch.py` 先验证可行）：左 padding 批量生成 `generate_many`；
  `num_return_sequences` 一次采 σN 个策略；rollout 锁步并行（每步批量生成所有活跃 episode 动作）；
  log-prob 批量前向；loss 批量前向+单次反向（B=2，受限于 152k 词表 lm_head logits 显存）。
  生成阶段实测 **~8-12× 加速**。

### 3.7 其它
- 输出缓冲：统一 `python3 -u`。
- QLoRA-4bit → bf16 LoRA（48GB 容得下 18GB 模型，更快、省去反量化开销）。
- warm-start：`init_adapter` 指向 SFT 产物，`PeftModel.from_pretrained(..., is_trainable=True)`。

---

## 4. 运行

### 4.1 环境
- Ubuntu 22.04，1× ~48GB GPU（实测占用 ~19-25GB），Python 3.10+。
- `torch`（建议与本机 CUDA 匹配，如 cu124/cu126）、`transformers`、`peft`、`trl`、`accelerate`、
  `datasets`、`numpy`。
- **强烈建议**让 `flash-linear-attention` + `causal-conv1d` 的快路径编译通过——否则 Gated DeltaNet
  回退到 torch 实现，生成只有 ~22 tok/s（详见「性能」）。
- 下载模型到 `model/`：
  ```bash
  huggingface-cli download empero-ai/Qwable-9B-Claude-Fable-5 --local-dir model
  ```

### 4.2 数据
```bash
python3 src/prepare_data.py      # 生成 data/train/* 与 data/eval/*
```
（HumanEval/MBPP 需联网；无网时自动只用合成任务。）

### 4.3 SFT 格式对齐
```bash
python3 -u src/sft_train.py       # -> checkpoints/sft
```

### 4.4 RL 训练（warm-start from SFT）
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u src/strata_trainer.py \
  --config configs/full.json \
  --train-data data/train/all_tasks.json \
  --eval-data   data/eval/eval_small.json
```
每 20 步存 `checkpoints/checkpoint-N` 与 `checkpoints/best`（按 eval 更新）。

### 4.5 评估
```bash
python3 -u src/eval.py --adapter checkpoints/sft --n 12 --max-steps 6 --data data/eval/eval_small.json
```

### 4.6 冒烟
```bash
python3 -u src/smoke_test.py                     # 极小步数端到端
python3 -u src/probe_batch.py                    # 验证批量生成 + 测速
```

---

## 5. 性能与已知局限

- **吞吐瓶颈是生成速度，不是显存**：本机 ~22 tok/s（fla 快路径未编译通过）→ 100 步约数小时。
  让快路径生效（干净的 CUDA/fla/causal-conv1d 栈）可达 ~100-150 tok/s，100 步压缩到 ~2h。
 详见对话中"什么样的机器能在 2 小时跑完"的分析。
- **RL 信号窄**：SFT 后模型 ~62.5%，简单任务全解出、难任务全失败 → 组内零方差 → 无梯度；
  真正可学习的只有中等难度那一带。`batch_size>1` + 温度采样可提升非零信号步占比。
- **安全提示**：`CodeGym` 用 `subprocess` 以 root 在宿主执行模型生成的命令，**无 Docker 隔离**，
  仅适用于一次性可丢弃的实验机。

---

## 6. 致谢 / 依据

- 基座模型：`empero-ai/Qwable-9B-Claude-Fable-5`（HuggingFace）。
- 方法：StraTA（Strategic Trajectory Abstraction）分层 agentic RL。
- 数据：HumanEval、MBPP（经 `datasets`），外加合成编码任务。

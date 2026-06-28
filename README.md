# StraTA on Qwable-9B-Claude-Fable-5 — 个人复现 / 测试

> **声明 / Attribution**
> 本仓库是对 **StraTA: Strategic Trajectory Abstraction for Agentic RL**（论文：[arXiv:2605.06642](https://arxiv.org/abs/2605.06642)）这一研究方法的**个人爱好者独立复现与测试**。**非官方实现，与论文原作者无任何关联。** 代码、超参、结果均为个人学习/实验产物，正确性与完整性不作任何保证；如需权威实现请参阅原论文。
>
> 基座模型为 [`empero-ai/Qwable-9B-Claude-Fable-5`](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5)（Qwen3.5-9B 系，混合 Gated DeltaNet 线性注意力 + `<think>` 推理，冻结视觉的多模态架构）。

本仓库记录了在单卡上把这条 **SFT 格式对齐 → GRPO 分层智能体强化学习** 管线从"完全跑不起来"调试到"端到端跑通"的完整过程。**真正的价值不在复现方法本身，而在定位并修复了一连串阻塞性 bug**（见「第 3 节 关键修复」）。

---

## 目录
1. [方法概览](#1-方法概览strata)
2. [仓库结构](#2-仓库结构)
3. [关键修复（从跑不通到端到端跑通）](#3-关键修复从跑不通到端到端跑通)
4. [运行（从零开始 / 干净机器）](#4-运行从零开始--干净机器)
5. [性能与已知局限](#5-性能与已知局限)
6. [安全提示](#6-安全提示)
7. [致谢](#7-致谢)

---

## 1. 方法概览（StraTA）

分层 agentic RL。一个训练 step 分五阶段（公式编号对应原论文）：

1. **策略采样**：用 `STRATEGY_PROMPT` 采样 `σ·N` 个候选策略，再用基于 sentence-embedding 的 **FPS（最远点采样）** 做多样性筛选，取 `N` 条（论文 Algorithm 1）。
2. **分层 Rollout**：每条策略在 `CodeGym` 沙盒里跑 `M` 个多步交互 episode。
3. **奖励计算**：策略级用 top-δ 聚合（论文 Eq. 10）；动作级用**跨 rollout 的 group-relative 优势（正确 GRPO，论文 Eq. 2）** + 长度惩罚（Eq. 11）+ 格式惩罚（Eq. 12）+ 自我评判（self-judgment）惩罚 κ。
4. **旧策略 log-prob 采集**（no_grad，批量化）。
5. **GRPO clipped-surrogate loss + KL 近似**（无参考模型，省 ~5GB 显存），批量化反向。

> 说明：本复现受个人算力约束，参数规模（N/M/σ）远小于论文设定；目标是跑通流程、验证信号，而非复现论文指标。

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
  smoke_test.py       # 端到端冒烟（极小 N/M/步数）
  quick_test.py       # 模型加载 + 显存自检
  probe_batch.py      # 探针：验证批量/left-pad 生成在自定义架构上可用 + 测速
configs/
  full.json           # 生产 RL 配置（warm-start SFT，batch_size=2，max_steps=100）
  smoke*.json         # 各阶段冒烟配置
data/
  train/              # all_tasks.json(364) synthetic.json humaneval.json sft_data.json(300)
  eval/               # eval_tasks.json(100) eval_small.json(12)
requirements.txt      # 固定版本的运行依赖
```

**所有路径都已改为相对项目根**（代码用 `__file__` 自动定位），在任意位置 clone、任意工作目录下都能跑。

> 模型权重（`model/`，~18GB）与训练产物（`checkpoints/`）不入库，按「4. 运行」步骤自行下载/生成。

---

## 3. 关键修复（从跑不通到端到端跑通）

复现最初完全跑不起来。按发现顺序，根因与修复如下：

### 3.1 chat-template 训练/推理不一致 → action 全空、0% 解出
- `generate()` 喂的是**原始文本**，而模型（Qwen3.5 系）训练分布是 chat 模板 → 输入 OOD，生成乱码/空。
- **修复**：`generate()` / log-prob / loss 全部先 `apply_chat_template(..., add_generation_prompt=True)` 再 tokenize；SFT 数据也按 per-prompt 协议重建。

### 3.2 `_collect_log_probs` 返回错误变量 → 训练崩溃
- 返回了未定义的 `log_probs` 而非累加列表。**修复**：返回 `log_probs_list`。

### 3.3 空目标 / 非有限 loss → NaN 腐蚀 LoRA 权重
- 空目标做 `.mean()` 产生 NaN 并反向。**修复**：空目标过滤 + `torch.isfinite` 守卫，非有限直接跳过该 minibatch 的 backward。

### 3.4 ★验证协议错配（决定性的解题率修复）
- 沙盒 `_check_success` 跑 `task["test_command"]` = `python3 -c "from solution import ..."`，因此模型**必须**写到 `solution.py`；但提示从没告诉它（SFT 数据是通用的，写 `main.py`）。模型把**正确的代码写进了错误的文件** → import 失败 → 永远 solved=False。
- **修复**：`augment_description(task)` 把验证命令 + "写入被 import 的模块（如 solution.py）" + 动作 DSL 追加到任务描述，在 `training_step` / `_run_episode` / `evaluate` / `eval.py` **多处一致**注入（避免再造训练/推理偏移）。
- **效果**（同一 SFT 模型，零额外训练）：eval_small 解题率 **12.5% → 62.5%**，动作格式从 0% 空变为真实代码。

### 3.5 ★GRPO 信用分配写错 → 训练无信号（Items=0）
- 原代码对**单个 rollout 内的时间步**算 group-relative 优势；而绝大多数任务第 1 步就解出（`len(rewards)==1`）→ `compute_advantages` 返回全 0 → 所有 item 被零优势过滤 → 不训练。
- **修复**：正确的 GRPO——优势在**同一策略的 M 个 rollout 之间**计算（group = M rollouts），每个 rollout 的所有动作继承该 rollout 的组相对优势。配合 `batch_size>1` 保证每步大概率有非零信号。

### 3.6 ★生成/rollout/log-prob 全串行 → 显存闲置、极慢
- 48GB 卡只用了 ~19GB，串行生成把卡浪费了 ~8-12×。
- **修复**（探针 `probe_batch.py` 先验证可行）：左 padding 批量生成 `generate_many`；`num_return_sequences` 一次采 σN 个策略；rollout 锁步并行（每步批量生成所有活跃 episode 动作）；log-prob 批量前向（B=16, no_grad）；loss 批量前向+单次反向（B=2，受限于 152k 词表 lm_head logits 显存）。生成阶段实测 **~8-12× 加速**。

### 3.7 ★生成 prompt ≠ log-prob prompt → GRPO 比率失真
- rollout 历史用的是完整 "Action/Result"，而 log-prob 用的是最近 3 步短历史 → 两阶段 prompt 不一致 → importance-sampling 比率失真。
- **修复**：rollout 改用 `_build_action_prompt`（最近 3 步短历史），并在生成时把**每一步的精确 prompt 存进 `trajectory.prompts`**，log-prob/loss 直接取用 → 生成 prompt == log-prob prompt == loss prompt。

### 3.8 硬编码绝对路径 → 换机器/换目录全跑不了
- 全部脚本与配置里写死了 `/root/strata-project/...`。
- **修复**：所有路径改为相对**项目根**（由 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 自动定位）；`strata_trainer.py` 的 `main()` 统一把相对路径（含 JSON 里的 `init_adapter`）解析到项目根；configs 的 `init_adapter` 改为 `checkpoints/sft`。**任意目录 clone 即可运行。**

### 3.9 其它
- 输出缓冲：统一 `python3 -u`。
- QLoRA-4bit → bf16 LoRA（48GB 容得下 18GB 模型，更快、省去反量化开销）。
- warm-start：`init_adapter` 指向 SFT 产物，`PeftModel.from_pretrained(..., is_trainable=True)`；若 SFT 产物不存在则自动回退到全新 LoRA（冒烟可不先做 SFT）。
- 固定依赖版本：新增 `requirements.txt`。

---

## 4. 运行（从零开始 / 干净机器）

> 以下命令默认在**仓库根目录**执行（`cd` 到 clone 出来的目录）。由于路径已相对化，从别处执行也能跑。

### 4.1 硬件 & 系统
- 1× GPU（实测占用 ~19-25GB；**24GB 卡可跑**，更大更从容）。Ubuntu 22.04，Python 3.10+。
- **想跑得快**：务必让 fla 快路径编译通过（见 4.2 末）。否则 Gated DeltaNet 回退到 torch 实现，生成只有 ~22 tok/s。

### 4.2 环境
```bash
# 1) 先装匹配你 CUDA 的 torch（参考机为 CUDA 12.4）：
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 2) 再装其余依赖（版本已固定）：
pip install -r requirements.txt

# 3) 【强烈建议】编译 fla 快路径（针对你的 torch+CUDA 从源码装，生成可达 ~100-150 tok/s）：
pip install flash-linear-attention==0.5.1
pip install causal-conv1d        # 没有匹配 wheel 时会从源码编译，需要 nvcc
```
> 若 `flash-linear-attention` / `causal-conv1d` 编译失败，模型**仍可运行**（torch 回退），只是慢。这两者能否编译是本机与新机器速度差距的根因。

### 4.3 模型
```bash
huggingface-cli download empero-ai/Qwable-9B-Claude-Fable-5 --local-dir model
```

### 4.4 数据
```bash
python3 -u src/prepare_data.py      # 生成 data/train/* 与 data/eval/*
```
（HumanEval/MBPP 需联网；无网时自动只用合成任务。）

### 4.5 SFT 格式对齐
```bash
python3 -u src/sft_train.py         # -> checkpoints/sft
```

### 4.6 RL 训练（warm-start from SFT）
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u src/strata_trainer.py \
  --config configs/full.json \
  --train-data data/train/all_tasks.json \
  --eval-data   data/eval/eval_small.json
```
每 20 步存 `checkpoints/checkpoint-N`（仅保留最近 2 个）与 `checkpoints/best`（按 eval 更新）；`logs/training_log.json` 记录每步指标。

### 4.7 评估
```bash
python3 -u src/eval.py --adapter checkpoints/sft --n 12 --max-steps 6 --data data/eval/eval_small.json
# 或用带显式验证协议注入的版本：
python3 -u src/eval_hint.py --adapter checkpoints/sft --n 12 --max-steps 6 --data data/eval/eval_small.json
```

### 4.8 冒烟 / 探针
```bash
python3 -u src/quick_test.py        # 仅加载模型 + 显存自检
python3 -u src/smoke_test.py        # 极小步数端到端（可不做 SFT，自动回退全新 LoRA）
python3 -u src/probe_batch.py       # 验证批量生成 + 测速
```

---

## 5. 性能与已知局限

- **吞吐瓶颈是生成速度，不是显存**：参考机 ~22 tok/s（fla 快路径未编译通过）→ 100 步约数小时。让快路径生效（干净的 CUDA/fla/causal-conv1d 栈）可达 ~100-150 tok/s，100 步可压缩到 ~2h。
- **RL 信号窄**：SFT 后模型 ~62.5%，简单任务全解出、难任务全失败 → 组内零方差 → 无梯度；真正可学习的只有中等难度那一带。`batch_size>1` + 温度采样可提升非零信号步占比。
- **RL 收敛未验证**：本仓库验证了训练管线跑通、产生真实梯度信号；**"训练后解题率是否提升"需在更强机器上跑完整训练后用 eval 确认**，本复现不保证。
- **个人复现性质**：超参与实现均简化，非论文级复现；仅供学习与流程验证。

---

## 6. 安全提示

⚠️ `CodeGym` 沙盒用 `subprocess` **以 root 在宿主机上直接执行模型生成的命令**，**无 Docker 隔离**。这仅适用于**一次性、可丢弃的实验机**。切勿在生产环境或含敏感数据的机器上运行。

---

## 7. 致谢

- **方法**：StraTA（Strategic Trajectory Abstraction）— [arXiv:2605.06642](https://arxiv.org/abs/2605.06642)。本仓库仅为个人复现，与原作者无关。
- **基座模型**：[`empero-ai/Qwable-9B-Claude-Fable-5`](https://huggingface.co/empero-ai/Qwable-9B-Claude-Fable-5)（HuggingFace）。
- **数据**：HumanEval、MBPP（经 `datasets`），外加合成编码任务。

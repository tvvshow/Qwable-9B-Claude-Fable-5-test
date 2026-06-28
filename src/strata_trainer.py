"""
StraTA: Strategic Trajectory Abstraction for Agentic RL
Core training framework adapted for code tasks.
Based on: arXiv:2605.06642

Memory-optimized for single RTX 4090 (48GB) + 9B model QLoRA 4-bit.
Key optimizations:
- No reference model (use KL approximation instead)
- Two-pass training: no_grad collection + incremental gradient computation
- Aggressive GPU memory management
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Configuration
# ============================================================
@dataclass
class StraTAConfig:
    """StraTA training configuration - tuned for single RTX 4090 (48GB) + Qwable-9B"""
    # Model
    model_path: str = "/root/strata-project/model"
    embedding_model: str = "all-MiniLM-L6-v2"
    load_4bit: bool = False   # 48GB box: bf16 LoRA is faster & avoids bnb dequant overhead
    init_adapter: Optional[str] = None   # SFT LoRA adapter to warm-start RL from

    # StraTA core params (from paper Table 7)
    N: int = 4              # strategies per task
    M: int = 4              # rollouts per strategy (reduced from 8 for single GPU)
    sigma: int = 4           # oversampling ratio (reduced from 8)
    delta: float = 0.5       # top-delta aggregation
    kappa: float = 0.1       # self-judgment penalty weight
    lam: float = 0.5         # length penalty threshold

    # Training
    learning_rate: float = 2e-6
    kl_beta: float = 0.01    # KL regularization coefficient
    batch_size: int = 1      # tasks per step (single GPU)
    max_steps: int = 300
    max_interaction_steps: int = 20
    train_temperature: float = 1.0
    eval_temperature: float = 0.7
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Token limits (reduced for memory)
    max_prompt_tokens: int = 4096
    max_response_tokens: int = 512

    # Paths
    checkpoint_dir: str = "/root/strata-project/checkpoints"
    data_dir: str = "/root/strata-project/data"
    eval_dir: str = "/root/strata-project/evals"
    log_dir: str = "/root/strata-project/logs"


# ============================================================
# 2. Prompt Templates
# ============================================================
STRATEGY_PROMPT = """{task_description}

当前项目状态：
{project_context}

在开始编码之前，请先制定一个全局策略。要求如下：
1. 策略为一段固定的文本，将指导你在整个任务周期中的行动方向。
2. 策略需要足够具体，使得后续每一步行动都能严格遵循它。
3. 策略需要足够可行，基于当前已知信息来制定。

先逐步思考整体规划，然后在 <strategy>...</strategy> 标签中给出你的策略。"""

ACTION_PROMPT = """{task_description}

你需要严格遵循以下全局策略来执行操作：
<strategy>
{strategy}
</strategy>

当前项目状态：
{current_state}

交互历史（最近步骤）：
{recent_history}

请执行下一步操作，使用 <action>...</action> 标签包裹你的操作。"""

SELF_JUDGMENT_PROMPT = """{task_description}

本次任务的全局策略如下：
<strategy>
{strategy}
</strategy>

完整的操作历史如下：
{action_history}

请指出所有存在问题的操作步骤。一个步骤被认为有问题，当且仅当它既不遵循全局策略，也没有推进任务进度。

先逐步分析历史，然后在 <judgment>...</judgment> 标签中给出有问题步骤的编号列表。
如果所有步骤都合理，输出空列表。
示例：<judgment>[2, 5]</judgment>"""


# Verification-protocol hint appended to a task description so the model knows the
# exact module/file the sandbox will import when checking success (e.g. solution.py).
# Injected consistently in training_step, _run_episode and evaluate so SFT-warmstarted
# RL trains on the SAME prompt distribution used at eval time (no train/inference skew).
VERIFY_HINT = (
    "\n\n[验证协议] 你的代码将通过以下命令自动验证，必须让它通过：\n"
    "{cmd}\n"
    "因此请把实现写入该命令 import 的模块文件（例如 `from solution import X` 则写入 solution.py）。\n"
    "可用动作格式：write:<文件名>\\n<文件内容>、read:<文件名>、bash:<命令>、test:<命令>。\n"
    "完成实现后，用 `test:{cmd}` 运行该验证命令确认通过。"
)


def augment_description(task: Dict) -> str:
    """Append the verification-protocol hint when the task carries a test_command."""
    desc = task["description"]
    cmd = task.get("test_command", "")
    return desc + VERIFY_HINT.format(cmd=cmd) if cmd else desc


# ============================================================
# 3. Diverse Strategy Sampler (Farthest Point Sampling)
# ============================================================
class DiverseStrategySampler:
    """FPS-based diverse strategy selection (Paper Algorithm 1)"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)

    def select(self, candidates: List[str], n_select: int) -> List[str]:
        """Select n_select maximally diverse strategies from candidates."""
        if len(candidates) <= n_select:
            return candidates

        embeddings = self.encoder.encode(candidates, normalize_embeddings=True)

        centroid = embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sims_to_centroid = embeddings @ centroid
        first_idx = int(np.argmax(sims_to_centroid))

        selected = [first_idx]
        selected_embs = [embeddings[first_idx]]

        while len(selected) < n_select:
            best_idx = -1
            best_min_sim = float("inf")
            for i in range(len(embeddings)):
                if i in selected:
                    continue
                max_sim = max(float(embeddings[i] @ se) for se in selected_embs)
                if max_sim < best_min_sim:
                    best_min_sim = max_sim
                    best_idx = i
            selected.append(best_idx)
            selected_embs.append(embeddings[best_idx])

        return [candidates[i] for i in selected]


# ============================================================
# 4. Reward Functions
# ============================================================
def compute_strategy_reward(rollout_rewards: List[float], delta: float = 0.5) -> float:
    """Top-delta aggregation for strategy reward (Paper Eq. 10)"""
    sorted_rewards = sorted(rollout_rewards, reverse=True)
    top_k = max(1, int(len(sorted_rewards) * delta))
    return float(np.mean(sorted_rewards[:top_k]))


def length_penalty(token_count: int, L_total: int, lam: float = 0.5) -> float:
    """Soft length penalty (Paper Eq. 11)"""
    threshold = int(lam * L_total)
    if token_count <= threshold:
        return 0.0
    elif token_count <= L_total:
        return -(token_count - threshold) / ((1 - lam) * L_total)
    else:
        return -1.0


def format_penalty(is_valid: bool) -> float:
    """Hard format penalty (Paper Eq. 12)"""
    return 0.0 if is_valid else -1.0


def compute_advantages(rewards: List[float]) -> List[float]:
    """Group-relative advantage (Paper Eq. 2)"""
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    if std_r < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean_r) / std_r for r in rewards]


# ============================================================
# 5. Utility Functions
# ============================================================
def extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag>...</tag>"""
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return text[start + len(f"<{tag}>"):end].strip()


def parse_judgment(text: str) -> List[int]:
    """Parse self-judgment output to list of step indices"""
    content = extract_tag(text, "judgment")
    if not content:
        return []
    import re
    numbers = re.findall(r'\d+', content)
    return [int(n) - 1 for n in numbers]


@dataclass
class Trajectory:
    """A single rollout trajectory"""
    strategy: str
    actions: List[str]
    observations: List[str]
    rewards: List[float]
    total_reward: float = 0.0
    token_counts: List[int] = field(default_factory=list)
    valid_formats: List[bool] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)  # exact action prompt seen at gen time

    def __len__(self):
        return len(self.actions)


# ============================================================
# 6. Core StraTA Training Loop (Memory-Optimized)
# ============================================================
class StraTATrainer:
    """Main StraTA training orchestrator.

    Memory optimization strategy:
    - NO reference model (saves ~5GB GPU). KL regularization uses a simple
      approximation: penalize deviation from initial log probs stored at setup.
    - Two-pass training_step:
      Pass 1 (no_grad): collect all prompts, actions, advantages
      Pass 2 (with_grad): compute loss incrementally, backward per item
    - Aggressive cache clearing between phases
    """

    def __init__(self, config: StraTAConfig):
        self.config = config
        self.sampler = DiverseStrategySampler(config.embedding_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.optimizer = None

    def _log_gpu(self, tag: str = ""):
        """Log GPU memory usage for debugging."""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  [GPU {tag}] Alloc: {alloc:.2f} GB, Reserved: {reserved:.2f} GB")

    def setup(self):
        """Initialize model, tokenizer, optimizer. NO reference model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding is required for correct batched generation (new tokens must
        # align at the right edge across variable-length prompts).
        self.tokenizer.padding_side = "left"

        if self.config.load_4bit:
            from transformers import BitsAndBytesConfig
            print("Loading model with QLoRA 4-bit...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=bnb_config,
                device_map="cuda",
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            print("Loading model in bf16 (LoRA, no quantization)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="cuda",
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )

        # Enable gradient checkpointing (+ make inputs require grad so LoRA grads flow)
        self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        print("  Gradient checkpointing enabled")

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        if self.config.init_adapter and os.path.isdir(self.config.init_adapter):
            from peft import PeftModel
            print(f"  Warm-starting LoRA from SFT adapter: {self.config.init_adapter}")
            self.model = PeftModel.from_pretrained(
                self.model, self.config.init_adapter, is_trainable=True
            )
        else:
            self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self._log_gpu("after model load")

        # NO reference model — saves ~5GB GPU VRAM
        # KL regularization uses log prob ratio clipping instead

        # Optimizer (only LoRA params)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        torch.cuda.empty_cache()
        self._log_gpu("after setup complete")
        print("Setup complete! (No reference model — memory optimized)")

    @torch.no_grad()
    def generate(self, prompt: str, temperature: float = None, max_new_tokens: int = None) -> str:
        """Generate text from model. Gradient checkpointing disabled during generation."""
        if temperature is None:
            temperature = self.config.train_temperature
        if max_new_tokens is None:
            max_new_tokens = self.config.max_response_tokens

        # Temporarily disable gradient checkpointing for generation
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

        # Wrap as a chat turn so input matches the model's training distribution
        # (and the SFT format-alignment data). Raw-text prompts are OOD for Qwen3.5.
        chat_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt", truncation=True,
                                max_length=self.config.max_prompt_tokens)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=20,
                do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        finally:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        return result

    def generate_strategy(self, task_description: str, project_context: str) -> str:
        """Generate a global strategy from initial state"""
        prompt = STRATEGY_PROMPT.format(
            task_description=task_description,
            project_context=project_context
        )
        response = self.generate(prompt, max_new_tokens=512)
        strategy = extract_tag(response, "strategy")
        return strategy if strategy else response[:200]

    def generate_action(self, task_description: str, strategy: str,
                       current_state: str, history: str) -> Tuple[str, int, bool]:
        """Generate an action conditioned on strategy and state"""
        prompt = ACTION_PROMPT.format(
            task_description=task_description,
            strategy=strategy,
            current_state=current_state,
            recent_history=history
        )
        response = self.generate(prompt, max_new_tokens=256)
        action = extract_tag(response, "action")
        token_count = len(self.tokenizer.encode(response))
        is_valid = action is not None
        return (action if action else response[:200], token_count, is_valid)

    def self_judge(self, task_description: str, strategy: str,
                   trajectory: Trajectory) -> List[int]:
        """Critical self-judgment: flag problematic steps"""
        history_text = ""
        for i, (action, obs) in enumerate(zip(trajectory.actions, trajectory.observations)):
            history_text += f"Step {i+1}: Action: {action}\nResult: {obs}\n"

        prompt = SELF_JUDGMENT_PROMPT.format(
            task_description=task_description,
            strategy=strategy,
            action_history=history_text
        )
        response = self.generate(prompt, max_new_tokens=256)
        return parse_judgment(response)

    def _build_action_prompt(self, task_desc, strategy, observation, actions_history):
        """Build the action prompt text."""
        return ACTION_PROMPT.format(
            task_description=task_desc,
            strategy=strategy,
            current_state=observation,
            recent_history=actions_history
        )

    # ================= Batched generation (uses idle VRAM for big speedups) =======
    def _chat(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

    @torch.no_grad()
    def generate_many(self, prompts: List[str], max_new_tokens: int,
                      temperature: float = None, num_return_sequences: int = 1) -> List[str]:
        """Generate for a batch of prompts in ONE forward pass (left-padded).

        Returns a flat list of length len(prompts)*num_return_sequences, ordered
        prompt-major (HF: [p0_s0, p0_s1, ..., p1_s0, ...]). The custom hybrid
        linear-attention arch handles batched/padded decoding (verified by probe).
        """
        if not prompts:
            return []
        if temperature is None:
            temperature = self.config.train_temperature
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        chats = [self._chat(p) for p in prompts]
        enc = self.tokenizer(chats, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.config.max_prompt_tokens,
                             add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        try:
            outputs = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=0.95, top_k=20, do_sample=True, repetition_penalty=1.05,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
            )
            gen = outputs[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        finally:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
        return texts

    def generate_strategies_batched(self, task_description: str, project_context: str,
                                    n_candidates: int) -> List[str]:
        """Sample n_candidates strategies from one prompt in a single batched call."""
        prompt = STRATEGY_PROMPT.format(task_description=task_description,
                                        project_context=project_context)
        texts = self.generate_many([prompt], max_new_tokens=512,
                                   num_return_sequences=n_candidates)
        return [extract_tag(t, "strategy") or (t[:200] if t else "(empty)") for t in texts]

    def run_episodes_batched(self, task: Dict, strategies: List[str], M: int):
        """Run len(strategies)*M episodes in lockstep, batching the action
        generation across all still-active episodes at each interaction step.
        Returns strategy_groups: List[(strategy, [Trajectory, ...])]."""
        from sandbox import CodeGym
        cfg = self.config
        task_desc = augment_description(task)
        n_eps = len(strategies) * M
        strat_of = [si for si in range(len(strategies)) for _ in range(M)]
        envs = [CodeGym(task, max_steps=cfg.max_interaction_steps) for _ in range(n_eps)]
        obs_list = [e.reset() for e in envs]
        trajs = [Trajectory(strategy=strategies[strat_of[i]], actions=[],
                            observations=[], rewards=[]) for i in range(n_eps)]
        done = [False] * n_eps

        for _step in range(cfg.max_interaction_steps):
            active = [i for i in range(n_eps) if not done[i]]
            if not active:
                break
            prompts = []
            for i in active:
                tr = trajs[i]
                t_now = len(tr.actions)
                # Identical history format to _build_action_prompt (last-3 short) so the
                # generation prompt == the log-prob prompt → correct GRPO ratios.
                history = "\n".join(f"Step {k+1}: {tr.actions[k]}"
                                    for k in range(max(0, t_now - 3), t_now))
                prompts.append(self._build_action_prompt(
                    task_desc, strategies[strat_of[i]], obs_list[i], history))
            responses = self.generate_many(prompts, max_new_tokens=256)
            for idx, i in enumerate(active):
                resp = responses[idx]
                action = extract_tag(resp, "action")
                is_valid = action is not None
                action = action if action else resp[:200]
                token_count = len(self.tokenizer.encode(resp))
                obs, reward, d = envs[i].step(action)
                tr = trajs[i]
                tr.actions.append(action)
                tr.observations.append(obs)
                tr.rewards.append(reward)
                tr.token_counts.append(token_count)
                tr.valid_formats.append(is_valid)
                tr.prompts.append(prompts[idx])  # exact prompt this action was sampled from
                obs_list[i] = obs
                if d:
                    done[i] = True

        for i in range(n_eps):
            trajs[i].total_reward = envs[i].get_total_reward()
        return [(strategies[si], [trajs[i] for i in range(n_eps) if strat_of[i] == si])
                for si in range(len(strategies))]

    def self_judge_batched(self, task_desc: str, strategy_groups) -> Dict:
        """Batched self-judgment over all rollouts. Returns {(g_idx, r_idx): [flagged steps]}."""
        if self.config.kappa <= 0:
            return {}
        prompts, index = [], []
        for g_idx, (strategy, rollouts) in enumerate(strategy_groups):
            for r_idx, rollout in enumerate(rollouts):
                if len(rollout) == 0:
                    continue
                history_text = "".join(
                    f"Step {i+1}: Action: {a}\nResult: {o}\n"
                    for i, (a, o) in enumerate(zip(rollout.actions, rollout.observations)))
                prompts.append(SELF_JUDGMENT_PROMPT.format(
                    task_description=task_desc, strategy=strategy,
                    action_history=history_text))
                index.append((g_idx, r_idx))
        if not prompts:
            return {}
        responses = self.generate_many(prompts, max_new_tokens=256)
        return {idx: parse_judgment(resp) for idx, resp in zip(index, responses)}

    def _encode_prompt_ids(self, prompt_text: str):
        """Chat-template a prompt and return its token ids (matches generate())."""
        chat_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(
            chat_text, return_tensors="pt",
            truncation=True, max_length=self.config.max_prompt_tokens,
            add_special_tokens=False,
        )["input_ids"]

    @torch.no_grad()
    def _collect_log_probs(self, items: List[Tuple[str, str]]) -> List[float]:
        """Pass 1: Collect mean target log-probs WITHOUT gradients, BATCHED.

        Left-pads each (prompt+target) sequence so every row is right-aligned; the
        last `target_len` logits of each row predict that row's target tokens. One
        forward per mini-batch instead of one per item (uses idle VRAM).
        """
        results = [0.0] * len(items)
        encoded = []
        for p_text, a_text in items:
            pid = self._encode_prompt_ids(p_text)[0]
            tid = self.tokenizer.encode(a_text, add_special_tokens=False,
                                        return_tensors="pt")[0]
            encoded.append((pid, tid))

        B = 16
        pad_id = self.tokenizer.pad_token_id
        for start in range(0, len(encoded), B):
            chunk = encoded[start:start + B]
            seqs = [torch.cat([pid, tid]) for pid, tid in chunk]
            maxlen = max(s.size(0) for s in seqs)
            input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for j, s in enumerate(seqs):
                input_ids[j, maxlen - s.size(0):] = s          # left-pad
                attn[j, maxlen - s.size(0):] = 1
            input_ids = input_ids.to(self.device)
            attn = attn.to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            for j, (pid, tid) in enumerate(chunk):
                tl = tid.size(0)
                if tl == 0:
                    continue
                row = logits[j, maxlen - tl - 1: maxlen - 1, :]
                lp = F.log_softmax(row.float(), dim=-1)
                tlp = lp.gather(-1, tid.to(self.device).unsqueeze(-1)).squeeze(-1)
                results[start + j] = tlp.mean().item()
            del logits
            torch.cuda.empty_cache()

        return results

    def _compute_loss_incremental(self, items: List[Tuple[str, str]],
                                   old_log_probs: List[float],
                                   advantages: List[float]) -> Tuple[torch.Tensor, dict]:
        """Pass 2: WITH gradients, BATCHED. Same left-pad slicing as
        _collect_log_probs (proven correct). For each item compute the GRPO
        clipped-surrogate + KL from its target logits; sum the per-item losses in
        the mini-batch and do ONE backward. Gradient checkpointing bounds memory,
        so this uses the idle VRAM instead of N serial fwd/bwd passes.
        """
        total_loss_val = 0.0
        total_kl = 0.0
        n_items = 0
        eps = 0.2
        pad_id = self.tokenizer.pad_token_id

        encoded = []
        for (p_text, a_text), adv in zip(items, advantages):
            pid = self._encode_prompt_ids(p_text)[0]
            tid = self.tokenizer.encode(a_text, add_special_tokens=False,
                                        return_tensors="pt")[0]
            if tid.size(0) == 0:
                continue
            encoded.append((pid, tid, float(adv)))

        # Small batch: the with_grad forward must RETAIN [B, maxlen, 152k-vocab]
        # logits for backward (log-prob pass is no_grad so it can use B=16 safely).
        B = 2
        for start in range(0, len(encoded), B):
            chunk = encoded[start:start + B]
            seqs = [torch.cat([pid, tid]) for pid, tid, _ in chunk]
            maxlen = max(s.size(0) for s in seqs)
            cb = len(seqs)
            input_ids = torch.full((cb, maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((cb, maxlen), dtype=torch.long)
            for j, s in enumerate(seqs):
                input_ids[j, maxlen - s.size(0):] = s
                attn[j, maxlen - s.size(0):] = 1
            input_ids = input_ids.to(self.device)
            attn = attn.to(self.device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits

            new_lps = []
            for j, (pid, tid, adv) in enumerate(chunk):
                tl = tid.size(0)
                row = logits[j, maxlen - tl - 1: maxlen - 1, :]
                lp = F.log_softmax(row.float(), dim=-1)
                tlp = lp.gather(-1, tid.to(self.device).unsqueeze(-1)).squeeze(-1)
                new_lps.append(tlp.mean())
            new_lp = torch.stack(new_lps)  # [cb], shares the logits graph

            old_lp = torch.tensor([old_log_probs[start + k] for k in range(cb)],
                                  device=self.device, dtype=torch.float32)
            advs = torch.tensor([c[2] for c in chunk], device=self.device, dtype=torch.float32)

            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advs
            surrogate = -torch.min(surr1, surr2)   # [cb]
            kl = (new_lp - old_lp).abs()           # [cb]
            loss = (surrogate + self.config.kl_beta * kl).sum()

            # Guard: never backward a non-finite loss (would corrupt LoRA weights)
            if torch.isfinite(loss):
                loss.backward()
                total_loss_val += float(surrogate.sum().detach())
                total_kl += float(kl.sum().detach())
                n_items += cb

            del logits, new_lp, ratio, surr1, surr2, surrogate, kl, loss
            torch.cuda.empty_cache()

        avg_kl = total_kl / max(n_items, 1)
        avg_loss = total_loss_val / max(n_items, 1)
        return avg_loss, {"kl": avg_kl, "n_updated": n_items}


    def training_step(self, tasks: List[Dict]) -> Dict[str, float]:
        """Single StraTA training step — memory optimized.

        Phase 1: Generate strategies and rollouts (no_grad)
        Phase 2: Compute rewards and advantages (no_grad, CPU)
        Phase 3: Collect old log probs (no_grad)
        Phase 4: Compute loss incrementally (with_grad, one item at a time)
        Phase 5: Optimizer step
        """
        torch.cuda.empty_cache()
        cfg = self.config

        # ========== Phase 1: Strategy sampling + Rollouts ==========
        all_items = []  # (prompt_text, action_text, advantage)
        all_advantages = []

        for task in tasks:
            task_desc = augment_description(task)
            project_ctx = task.get("context", "Empty project")

            # Strategy sampling (batched: sigma*N candidates from a single call)
            n_candidates = cfg.sigma * cfg.N
            candidates = self.generate_strategies_batched(task_desc, project_ctx, n_candidates)
            strategies = self.sampler.select(candidates, cfg.N)

            # Hierarchical rollout (batched lockstep across all N*M episodes)
            strategy_groups = self.run_episodes_batched(task, strategies, cfg.M)

            # ========== Phase 2: Reward computation (CPU) ==========
            strategy_rewards = []
            for strategy, rollouts in strategy_groups:
                rollout_rewards = [r.total_reward for r in rollouts]
                s_reward = compute_strategy_reward(rollout_rewards, cfg.delta)
                s_len = len(self.tokenizer.encode(strategy))
                s_reward += length_penalty(s_len, cfg.max_response_tokens, cfg.lam)
                s_reward += format_penalty(extract_tag(strategy, "strategy") is not None)
                strategy_rewards.append(np.clip(s_reward, -1, 1))

            strategy_advantages = compute_advantages(strategy_rewards)

            # ===== Action-level GRPO: advantage is GROUP-RELATIVE ACROSS the M rollouts
            # of a strategy (proper GRPO baseline), NOT within a single rollout's steps
            # (which is ~always zero for short/1-step episodes). Each action inherits its
            # trajectory's group-relative advantage. =====
            flagged_map = self.self_judge_batched(task_desc, strategy_groups)
            for g_idx, (strategy, rollouts) in enumerate(strategy_groups):
                # Per-rollout scalar reward = terminal reward + mean per-action shaping
                rollout_scalars = []
                for r_idx, rollout in enumerate(rollouts):
                    flagged = flagged_map.get((g_idx, r_idx), [])
                    shaping = 0.0
                    for t in range(len(rollout)):
                        shaping += length_penalty(rollout.token_counts[t],
                                                  cfg.max_response_tokens, cfg.lam)
                        shaping += format_penalty(rollout.valid_formats[t])
                        if t in flagged:
                            shaping -= cfg.kappa
                    n = max(1, len(rollout))
                    R = rollout.total_reward + shaping / n
                    rollout_scalars.append(float(np.clip(R, -1, 1)))

                # Group-relative advantage across the M rollouts (nonzero iff outcomes differ)
                rollout_advs = compute_advantages(rollout_scalars)

                for r_idx, rollout in enumerate(rollouts):
                    adv = rollout_advs[r_idx]
                    for t, action in enumerate(rollout.actions):
                        # Use the EXACT prompt the action was sampled from (stored at gen
                        # time) → generation prompt == log-prob prompt == loss prompt.
                        prompt_text = rollout.prompts[t] if t < len(rollout.prompts) else \
                            self._build_action_prompt(task_desc, strategy,
                                                      rollout.observations[t] if t < len(rollout.observations) else "", "")
                        all_items.append((prompt_text, action))
                        all_advantages.append(adv)

                # Strategy items
                strat_prompt = STRATEGY_PROMPT.format(
                    task_description=task_desc, project_context=project_ctx
                )
                all_items.append((strat_prompt, strategy))
                all_advantages.append(strategy_advantages[g_idx])

        if not all_items:
            return {"loss": 0.0, "kl": 0.0, "mean_advantage": 0.0}

        # Filter out (a) items whose target encodes to 0 tokens (empty action/strategy,
        # would NaN the log-prob slice) and (b) zero-advantage items (no gradient signal).
        # Dropping zero-adv items here — not just in the loss pass — avoids wasting batched
        # log-prob forwards on them, which is the bulk of the saturated-step cost.
        filtered_items, filtered_advs = [], []
        for (p_text, a_text), adv in zip(all_items, all_advantages):
            if a_text and abs(adv) > 1e-8 and len(self.tokenizer.encode(a_text)) > 0:
                filtered_items.append((p_text, a_text))
                filtered_advs.append(adv)
        all_items, all_advantages = filtered_items, filtered_advs
        if not all_items:
            return {"loss": 0.0, "kl": 0.0, "mean_advantage": 0.0}

        # ========== Phase 3: Free GPU memory before training ==========
        torch.cuda.empty_cache()
        self._log_gpu("before log prob collection")

        # ========== Phase 4a: Collect old log probs (no_grad) ==========
        print(f"  Collecting log probs for {len(all_items)} items (no_grad)...")
        old_log_probs = self._collect_log_probs(all_items)

        torch.cuda.empty_cache()
        self._log_gpu("after log prob collection")

        # ========== Phase 4b: Compute loss incrementally (with_grad) ==========
        print(f"  Computing loss incrementally (with_grad)...")
        avg_loss, metrics = self._compute_loss_incremental(
            all_items, old_log_probs, all_advantages
        )

        # ========== Phase 5: Optimizer step ==========
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        mean_adv = float(np.mean(all_advantages)) if all_advantages else 0.0

        return {
            "loss": avg_loss,
            "kl": metrics["kl"],
            "mean_advantage": mean_adv,
            "n_items": len(all_items),
            "n_updated": metrics.get("n_updated", 0),
        }

    def _run_episode(self, task: Dict, strategy: str) -> Trajectory:
        """Run a single episode with strategy guidance"""
        from sandbox import CodeGym

        env = CodeGym(task, max_steps=self.config.max_interaction_steps)
        obs = env.reset()
        traj = Trajectory(strategy=strategy, actions=[], observations=[], rewards=[])

        for step in range(self.config.max_interaction_steps):
            history = "\n".join([
                f"Step {i+1}: Action: {a}\nResult: {o}"
                for i, (a, o) in enumerate(zip(traj.actions, traj.observations))
            ])

            action, token_count, is_valid = self.generate_action(
                augment_description(task), strategy, obs, history
            )

            obs, reward, done = env.step(action)
            traj.actions.append(action)
            traj.observations.append(obs)
            traj.rewards.append(reward)
            traj.token_counts.append(token_count)
            traj.valid_formats.append(is_valid)

            if done:
                break

        traj.total_reward = env.get_total_reward()
        return traj

    def train(self, train_data: List[Dict], eval_data: List[Dict] = None):
        """Main training loop"""
        cfg = self.config
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        print(f"Starting StraTA training: {cfg.max_steps} steps, "
              f"N={cfg.N}, M={cfg.M}, sigma={cfg.sigma}")
        print(f"Model: {cfg.model_path}")
        print(f"Training data: {len(train_data)} tasks")

        logs = []
        best_eval_score = -1

        for step in range(cfg.max_steps):
            batch = random.sample(train_data, min(cfg.batch_size, len(train_data)))

            print(f"\n--- Step {step}/{cfg.max_steps} ---")
            metrics = self.training_step(batch)
            metrics["step"] = step

            print(f"  Loss: {metrics['loss']:.4f} | "
                  f"KL: {metrics['kl']:.4f} | "
                  f"Adv: {metrics['mean_advantage']:.4f} | "
                  f"Items: {metrics.get('n_items', 0)}")

            logs.append(metrics)

            # Save checkpoint
            if (step + 1) % 20 == 0:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"checkpoint-{step+1}")
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")
                self._cleanup_checkpoints(keep=2)

            # Evaluation
            if eval_data and (step + 1) % 20 == 0:
                eval_score = self.evaluate(eval_data)
                metrics["eval_score"] = eval_score
                print(f"  Eval score: {eval_score:.4f}")
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    self.model.save_pretrained(
                        os.path.join(cfg.checkpoint_dir, "best")
                    )
                    print(f"  New best model saved!")

            # Save logs
            with open(os.path.join(cfg.log_dir, "training_log.json"), "w") as f:
                json.dump(logs, f, indent=2)

        print(f"\nTraining complete! Best eval score: {best_eval_score:.4f}")

    def evaluate(self, eval_data: List[Dict]) -> float:
        """Evaluate model on eval set"""
        from sandbox import CodeGym

        successes = 0
        for task in eval_data:
            strategy = self.generate_strategy(
                augment_description(task),
                task.get("context", "Empty project")
            )
            env = CodeGym(task, max_steps=self.config.max_interaction_steps)
            obs = env.reset()
            for step in range(self.config.max_interaction_steps):
                history = "\n".join([
                    f"Step {i+1}: {a}"
                    for a in env.last_actions[-5:]
                ]) if hasattr(env, 'last_actions') else ""
                action, _, _ = self.generate_action(
                    augment_description(task), strategy, obs, history
                )
                obs, reward, done = env.step(action)
                if done:
                    break
            if env.get_total_reward() > 0:
                successes += 1

        return successes / len(eval_data) if eval_data else 0.0

    def _cleanup_checkpoints(self, keep: int = 2):
        """Remove old checkpoints, keep only latest N"""
        import shutil
        ckpts = []
        for name in os.listdir(self.config.checkpoint_dir):
            path = os.path.join(self.config.checkpoint_dir, name)
            if os.path.isdir(path) and name.startswith("checkpoint-"):
                step = int(name.split("-")[1])
                ckpts.append((step, path))
        ckpts.sort(reverse=True)
        for step, path in ckpts[keep:]:
            shutil.rmtree(path, ignore_errors=True)
            print(f"  Cleaned up old checkpoint: {path}")


# ============================================================
# 7. Entry Point
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="StraTA Training")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--train-data", type=str, required=True, help="Training data JSON")
    parser.add_argument("--eval-data", type=str, default=None, help="Eval data JSON")
    args = parser.parse_args()

    config = StraTAConfig()
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

    with open(args.train_data) as f:
        train_data = json.load(f)
    eval_data = None
    if args.eval_data:
        with open(args.eval_data) as f:
            eval_data = json.load(f)

    trainer = StraTATrainer(config)
    trainer.setup()
    trainer.train(train_data, eval_data)


if __name__ == "__main__":
    main()

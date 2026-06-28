#!/usr/bin/env python3
"""
SFT format-alignment for StraTA on Qwable-9B (protocol-matched).

Builds two example types that mirror the EXACT prompts used at RL/eval time:
  - strategy:  STRATEGY_PROMPT(task)                  -> <strategy>...</strategy>
  - action:    ACTION_PROMPT(task, strategy, state..) -> <action>...</action>
Chat-templated prompt + assistant-only loss. Output: LoRA adapter at checkpoints/sft.
"""
import os, json, random, argparse, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from strata_trainer import STRATEGY_PROMPT, ACTION_PROMPT, extract_tag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "model")
SFT_DATA = os.path.join(ROOT, "data/train/sft_data.json")
OUT = os.path.join(ROOT, "checkpoints/sft")
MAXLEN = 3072
DEFAULT_CTX = "Empty project directory."


def parse_sample(messages):
    """Return (user_task, strategy_full, strategy_inner, [action_full...], [action_inner...])."""
    user_task = next((m["content"] for m in messages if m["role"] == "user"), "")
    strat_full, strat_inner = None, None
    actions_full, actions_inner = [], []
    for m in messages:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        if "<strategy>" in c and strat_full is None:
            strat_full, strat_inner = c.strip(), (extract_tag(c, "strategy") or "")
        elif "<action>" in c:
            actions_full.append(c.strip())
            actions_inner.append(extract_tag(c, "action") or "")
    return user_task, strat_full, strat_inner, actions_full, actions_inner


def build_examples(data):
    """Yield (prompt_text, target_text) pairs matching inference-time prompts."""
    out = []
    for d in data:
        user, strat_full, strat_inner, acts_full, acts_inner = parse_sample(d["messages"])
        if not user:
            continue
        if strat_full:
            out.append((STRATEGY_PROMPT.format(task_description=user, project_context=DEFAULT_CTX),
                        strat_full))
        for t, act_full in enumerate(acts_full):
            history = "\n".join(f"Step {i+1}: {acts_inner[i]}" for i in range(max(0, t - 3), t))
            state = f"[Step {t}] Project initialized. Files written so far: {t}."
            prompt = ACTION_PROMPT.format(
                task_description=user, strategy=strat_inner or "(see plan)",
                current_state=state, recent_history=history or "(none)",
            )
            out.append((prompt, act_full))
    return out


def tokenize(tok, prompt_text, target_text):
    chat = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                   tokenize=False, add_generation_prompt=True)
    pids = tok(chat, add_special_tokens=False)["input_ids"]
    tids = tok(target_text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (pids + tids)[-MAXLEN:]
    labels = ([-100] * len(pids) + tids)[-MAXLEN:]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    args = ap.parse_args()

    print("Loading tokenizer + model (bf16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.train()

    with open(SFT_DATA) as f:
        data = json.load(f)
    pairs = build_examples(data)
    examples = [tokenize(tok, p, t) for p, t in pairs]
    n_strat = sum(1 for p, _ in pairs if p.startswith(STRATEGY_PROMPT[:20]))
    print(f"SFT examples: {len(examples)} (strategy~{n_strat}, action~{len(examples)-n_strat}) "
          f"| VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    total = int(len(examples) * args.epochs)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * s / max(1, total)))))

    dev = torch.device("cuda")
    step, running = 0, 0.0
    opt.zero_grad()
    for epoch in range(math.ceil(args.epochs)):
        random.Random(epoch).shuffle(examples)
        for ids, labels in examples:
            if step >= total:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=torch.tensor([ids], device=dev),
                            labels=torch.tensor([labels], device=dev))
                loss = out.loss / args.accum
            loss.backward()
            running += out.loss.item()
            step += 1
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if step % 100 == 0:
                print(f"step {step}/{total} | loss {running/100:.4f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB",
                      flush=True)
                running = 0.0
        if step >= total:
            break

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"SFT DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

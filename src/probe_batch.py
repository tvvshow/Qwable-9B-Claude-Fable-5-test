#!/usr/bin/env python3
"""Probe: does batched (left-padded) generation + num_return_sequences work on
the custom Qwable hybrid-linear-attention arch? If yes, we can batch rollouts to
use idle VRAM and massively cut wall-clock. Prints per-call timing + VRAM."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "model")
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
model.eval()


def chat(p):
    return tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True)


prompts = [
    "用一句话说明什么是快速排序。",
    "写一个 Python 函数计算阶乘。",
    "解释二分查找的时间复杂度。",
]

# --- Serial baseline ---
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for p in prompts:
    ids = tok(chat(p), return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=64, do_sample=True, temperature=0.7,
                             pad_token_id=tok.pad_token_id)
serial_t = time.time() - t0
print(f"[SERIAL] 3 prompts x 64 tok: {serial_t:.1f}s  peakVRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

# --- Batched (left-padded) ---
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
enc = tok([chat(p) for p in prompts], return_tensors="pt", padding=True,
          add_special_tokens=False).to("cuda")
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.7,
                         pad_token_id=tok.pad_token_id)
batch_t = time.time() - t0
texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"[BATCH-3] {batch_t:.1f}s  peakVRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB  speedup {serial_t/batch_t:.1f}x", flush=True)
for i, t in enumerate(texts):
    print(f"  out[{i}]: {t[:70]!r}")

# --- num_return_sequences (same prompt, 8 samples) = strategy candidates ---
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
enc = tok(chat("为'实现一个回文检测函数'制定一个简短策略。"), return_tensors="pt",
          add_special_tokens=False).to("cuda")
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.9,
                         num_return_sequences=8, pad_token_id=tok.pad_token_id)
nrs_t = time.time() - t0
print(f"[NUM_RET=8] 8 samples x 64 tok: {nrs_t:.1f}s  peakVRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
print(f"  (serial-equiv would be ~{serial_t/3*8:.1f}s)  shapes ok: {out.shape}")

# --- Big batch 16 to probe VRAM headroom ---
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
enc = tok([chat(prompts[i % 3]) for i in range(16)], return_tensors="pt", padding=True,
          add_special_tokens=False).to("cuda")
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=128, do_sample=True, temperature=0.7,
                         pad_token_id=tok.pad_token_id)
big_t = time.time() - t0
print(f"[BATCH-16 x128tok] {big_t:.1f}s  peakVRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
print("PROBE OK", flush=True)

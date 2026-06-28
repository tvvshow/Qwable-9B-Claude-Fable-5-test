#!/usr/bin/env python3
"""Test the MERGED standalone model (base+best fused) for usability/correctness/accuracy.
Loads merged/ directly (zero-init LoRA no-op preserves merged weights), runs eval_small
through the CodeGym sandbox with the verification-hint protocol. No training."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_trainer import StraTAConfig, StraTATrainer
from sandbox import CodeGym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(ROOT, "merged")

HINT = (
    "\n\n[验证协议] 你的代码将通过以下命令自动验证，必须让它通过：\n{cmd}\n"
    "因此请把实现写入该命令 import 的模块文件（例如 `from solution import X` 则写入 solution.py）。\n"
    "可用动作格式：write:<文件名>\\n<文件内容>、read:<文件名>、bash:<命令>、test:<命令>。\n"
    "完成实现后，用 `test:{cmd}` 运行该验证命令确认通过。"
)
def aug(task):
    cmd = task.get("test_command", "")
    return task["description"] + (HINT.format(cmd=cmd) if cmd else "")

def main():
    assert os.path.isdir(MERGED), "merged model not found: " + MERGED
    print("=== Loading MERGED model: " + MERGED + " ===", flush=True)
    t0 = time.time()
    cfg = StraTAConfig(model_path=MERGED, init_adapter=None,
                       max_interaction_steps=6, eval_temperature=0.7)
    trainer = StraTATrainer(cfg)
    trainer.setup()
    print("  loaded in %ds | zero-init LoRA no-op -> merged weights tested faithfully\n" % (time.time()-t0), flush=True)

    with open(os.path.join(ROOT, "data/eval/eval_small.json")) as f:
        tasks = json.load(f)

    succ, fmt_ok, fmt_total = 0, 0, 0
    results = []
    for i, task in enumerate(tasks):
        desc = aug(task)
        t1 = time.time()
        tid = task.get("id", "?")
        strat = trainer.generate_strategy(desc, task.get("context", "Empty project"))
        env = CodeGym(task, max_steps=6)
        obs = env.reset()
        actions = []
        for _ in range(6):
            hist = "\n".join("Step %d: %s" % (j+1, a) for j, a in enumerate(actions[-4:]))
            action, _, is_valid = trainer.generate_action(desc, strat, obs, hist)
            fmt_total += 1; fmt_ok += int(is_valid)
            actions.append(action)
            obs, _, done = env.step(action)
            if done: break
        ok = env.get_total_reward() > 0
        succ += int(ok)
        dt = time.time() - t1
        results.append((tid, ok, len(actions), bool(strat)))
        print("[%d/%d] %-16s solved=%s steps=%d strat_fmt=%s %ds" %
              (i+1, len(tasks), tid, ok, len(actions), "Y" if strat else "N", int(dt)), flush=True)
        if i < 2:
            print("  --- STRATEGY ---")
            print("  " + (strat[:280] if strat else "(empty)").replace("\n", "\n  "))
            print("  --- ACTION[0] ---")
            print("  " + (actions[0][:360] if actions else "(none)").replace("\n", "\n  "))

    print("=" * 55, flush=True)
    print("MERGED MODEL TEST RESULTS (%d tasks, eval_small)" % len(tasks), flush=True)
    print("  CORRECTNESS (solve rate): %d/%d = %.1f%%" % (succ, len(tasks), succ/len(tasks)*100), flush=True)
    print("  USABILITY  (action format valid): %d/%d = %.1f%%" % (fmt_ok, max(fmt_total,1), fmt_ok/max(fmt_total,1)*100), flush=True)
    solved = [r[0] for r in results if r[1]]
    failed = [r[0] for r in results if not r[1]]
    print("  SOLVED:  %s" % solved, flush=True)
    print("  FAILED:  %s" % failed, flush=True)

if __name__ == "__main__":
    main()

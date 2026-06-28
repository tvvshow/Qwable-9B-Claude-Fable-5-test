#!/usr/bin/env python3
"""Eval with verification-protocol hint injected into task description.
Tests whether exposing the test_command lets the model target the right module
(e.g. solution.py) and run the test. Usage: eval_hint.py [--adapter P] [--n N] [--max-steps M]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_trainer import StraTAConfig, StraTATrainer
from sandbox import CodeGym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HINT = (
    "\n\n[验证协议] 你的代码将通过以下命令自动验证，必须让它通过：\n"
    "{cmd}\n"
    "因此请把实现写入该命令 import 的模块文件（例如 `from solution import X` 则写入 solution.py）。\n"
    "可用动作格式：write:<文件名>\\n<文件内容>、read:<文件名>、bash:<命令>、test:<命令>。\n"
    "完成实现后，用 `test:{cmd}` 运行该验证命令确认通过。"
)


def aug(task):
    cmd = task.get("test_command", "")
    return task["description"] + (HINT.format(cmd=cmd) if cmd else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--data", default=os.path.join(ROOT, "data/eval/eval_small.json"))
    args = ap.parse_args()

    cfg = StraTAConfig(init_adapter=args.adapter, max_interaction_steps=args.max_steps,
                       eval_temperature=0.7)
    trainer = StraTATrainer(cfg)
    trainer.setup()

    with open(args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)) as f:
        tasks = json.load(f)[: args.n]

    succ, fmt_ok, fmt_total = 0, 0, 0
    for i, task in enumerate(tasks):
        desc = aug(task)
        strat = trainer.generate_strategy(desc, task.get("context", "Empty project"))
        env = CodeGym(task, max_steps=args.max_steps)
        obs = env.reset()
        actions = []
        for _ in range(args.max_steps):
            hist = "\n".join(f"Step {j+1}: {a}" for j, a in enumerate(actions[-4:]))
            action, _, is_valid = trainer.generate_action(desc, strat, obs, hist)
            fmt_total += 1
            fmt_ok += int(is_valid)
            actions.append(action)
            obs, _, done = env.step(action)
            if done:
                break
        ok = env.get_total_reward() > 0
        succ += int(ok)
        print(f"[{i+1}/{len(tasks)}] {task['id']:<16} solved={ok} steps={len(actions)} "
              f"strat_fmt={'Y' if strat else 'N'}", flush=True)
        if i < 2:
            print("  --- sample action[0] ---\n  " +
                  (actions[0] if actions else "")[:300].replace("\n", "\n  "))

    print("=" * 50)
    print(f"adapter={args.adapter}  (WITH verification hint)")
    print(f"SUCCESS RATE: {succ}/{len(tasks)} = {succ/len(tasks)*100:.1f}%")
    print(f"ACTION FORMAT VALID: {fmt_ok}/{fmt_total} = {fmt_ok/max(fmt_total,1)*100:.1f}%")


if __name__ == "__main__":
    main()

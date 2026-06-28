#!/usr/bin/env python3
"""Quick eval: success-rate + format-validity on CodeGym eval tasks.
Usage: eval.py [--adapter PATH] [--n 10] [--max-steps 6]
"""
import sys, json, argparse
sys.path.insert(0, "/root/strata-project/src")
from strata_trainer import StraTAConfig, StraTATrainer, extract_tag, augment_description
from sandbox import CodeGym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit = base model)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--data", default="/root/strata-project/data/eval/eval_tasks.json")
    args = ap.parse_args()

    cfg = StraTAConfig(init_adapter=args.adapter, max_interaction_steps=args.max_steps,
                       eval_temperature=0.7)
    trainer = StraTATrainer(cfg)
    trainer.setup()

    with open(args.data) as f:
        tasks = json.load(f)[: args.n]

    succ, fmt_ok, fmt_total = 0, 0, 0
    for i, task in enumerate(tasks):
        desc = augment_description(task)
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
        print(f"[{i+1}/{len(tasks)}] {task['id']:<16} solved={ok} "
              f"steps={len(actions)} strat_fmt={'Y' if strat else 'N'}", flush=True)
        if i == 0:
            print("  --- sample strategy ---\n  " + (strat or "")[:300].replace("\n", "\n  "))
            print("  --- sample action[0] ---\n  " + (actions[0] if actions else "")[:300].replace("\n", "\n  "))

    print("=" * 50)
    print(f"adapter={args.adapter}")
    print(f"SUCCESS RATE: {succ}/{len(tasks)} = {succ/len(tasks)*100:.1f}%")
    print(f"ACTION FORMAT VALID: {fmt_ok}/{fmt_total} = {fmt_ok/max(fmt_total,1)*100:.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the trained StraTA coding agent on a custom task — this model's "CLI".

Loads the merged model (from HuggingFace by default, or a local dir), then runs
the full agent loop the model was trained for: emit a <strategy>, then loop
<action> steps (write:/read:/bash:/test:) inside a CodeGym sandbox until the
verification command passes or max-steps is hit. The whole interaction is printed.

This is the intended way to USE the model — it is an agentic coder bound to this
strategy/action + sandbox protocol, NOT a general chat model.

Examples:
  # default built-in task (two_sum)
  python src/run_task.py
  # custom task from the command line
  python src/run_task.py \
      --desc "implement factorial(n) returning n!" \
      --test 'python3 -c "from solution import factorial; assert factorial(5)==120; print(\"PASS\")"'
  # use a local merged dir instead of downloading from HF
  python src/run_task.py --model ./merged --desc "..." --test "..."

⚠ SECURITY: the CodeGym sandbox runs the model's bash/test actions as subprocesses
in the current working directory with NO docker isolation. Only run on a throwaway
machine / container, and never feed untrusted tasks.

Requires: torch, transformers, peft, flash-linear-attention (+ causal-conv1d for the
fast path). See requirements.txt. Needs >=24GB VRAM (bf16, 18GB model).
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_trainer import StraTAConfig, StraTATrainer
from sandbox import CodeGym

DEFAULT_MODEL = "pestlee/Qwable-9B-Claude-Fable-5-StraTA"

# Verification-protocol hint (same one used at train/eval time). The sandbox runs
# `from solution import ...`, so the model must write to the imported module file.
HINT = (
    "\n\n[验证协议] 你的代码将通过以下命令自动验证，必须让它通过：\n{cmd}\n"
    "因此请把实现写入该命令 import 的模块文件（例如 `from solution import X` 则写入 solution.py）。\n"
    "可用动作格式：write:<文件名>\\n<文件内容>、read:<文件名>、bash:<命令>、test:<命令>。\n"
    "完成实现后，用 `test:{cmd}` 运行该验证命令确认通过。"
)

DEFAULT_TASK = {
    "id": "two_sum",
    "description": "Implement `two_sum(nums, target)` that returns a list of the two indices "
                   "whose values add up to target. Assume exactly one solution exists.",
    "test_command": 'python3 -c "from solution import two_sum; '
                    'assert two_sum([2,7,11,15],9)==[0,1]; '
                    'assert two_sum([3,2,4],6)==[1,2]; '
                    'print(\'PASS\')"',
    "context": "Empty project",
    "files": {},
    "test_files": {},
    "difficulty": "easy",
    "max_steps": 8,
}


def aug(desc, cmd):
    return desc + (HINT.format(cmd=cmd) if cmd else "")


def main():
    ap = argparse.ArgumentParser(description="Run the StraTA coding agent on a task.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="merged model: HF repo id or local dir (default: %(default)s)")
    ap.add_argument("--desc", default=None, help="task description (overrides default task)")
    ap.add_argument("--test", default=None, help="verification command (overrides default)")
    ap.add_argument("--max-steps", type=int, default=8, help="max agent interaction steps")
    ap.add_argument("--temp", type=float, default=0.7, help="sampling temperature")
    args = ap.parse_args()

    task = dict(DEFAULT_TASK)
    if args.desc:
        task["description"] = args.desc
        task["id"] = "custom"
    if args.test:
        task["test_command"] = args.test
    task["max_steps"] = args.max_steps

    bar = "=" * 70
    print(bar, flush=True)
    print("MODEL:", args.model, flush=True)
    print("TASK:", task["id"], flush=True)
    print("DESC:", task["description"].strip(), flush=True)
    print("TEST:", task.get("test_command", ""), flush=True)
    print(bar, flush=True)

    print("\n[loading model...]", flush=True)
    t0 = time.time()
    cfg = StraTAConfig(model_path=args.model, init_adapter=None,
                       max_interaction_steps=args.max_steps, eval_temperature=args.temp)
    trainer = StraTATrainer(cfg)
    trainer.setup()
    print("[model ready in %ds]\n" % int(time.time() - t0), flush=True)

    desc = aug(task["description"], task.get("test_command", ""))

    print(">>> [STRATEGY]", flush=True)
    strat = trainer.generate_strategy(desc, task.get("context", "Empty project"))
    print(strat if strat else "(empty)", flush=True)
    print(flush=True)

    env = CodeGym(task, max_steps=args.max_steps)
    obs = env.reset()
    print(">>> [SANDBOX INIT]", obs, flush=True)
    print(flush=True)

    actions = []
    for step in range(args.max_steps):
        hist = "\n".join("Step %d: %s" % (j + 1, a) for j, a in enumerate(actions[-4:]))
        action, _, is_valid = trainer.generate_action(desc, strat, obs, hist)
        actions.append(action)
        print(">>> [STEP %d · ACTION] (valid=%s)" % (step + 1, is_valid), flush=True)
        print(action if action else "(empty)", flush=True)
        print(flush=True)
        obs, reward, done = env.step(action)
        print(">>> [SANDBOX]", flush=True)
        print(obs, flush=True)
        print("    (reward=%.2f, done=%s)" % (reward, done), flush=True)
        print(flush=True)
        if done:
            break

    ok = env.get_total_reward() > 0
    print(bar, flush=True)
    print(">>> VERDICT: %s  (steps=%d)" % ("✅ SOLVED" if ok else "❌ NOT SOLVED", len(actions)), flush=True)
    print(bar, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

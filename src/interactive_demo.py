#!/usr/bin/env python3
"""Interactive demo: load the MERGED model in the VPS and run it as a live coding agent.
Prints the FULL interaction trace (task -> strategy -> each action + sandbox feedback ->
final test verdict) so you can SEE the trained model actually working/interacting."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_trainer import StraTAConfig, StraTATrainer
from sandbox import CodeGym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(ROOT, "merged")

HINT = ("\n\n[验证协议] 你的代码将通过以下命令自动验证，必须让它通过：\n{cmd}\n"
        "因此请把实现写入该命令 import 的模块文件（例如 `from solution import X` 则写入 solution.py）。\n"
        "可用动作格式：write:<文件名>\\n<文件内容>、read:<文件名>、bash:<命令>、test:<命令>。\n"
        "完成实现后，用 `test:{cmd}` 运行该验证命令确认通过。")
def aug(task):
    cmd = task.get("test_command", "")
    return task["description"] + (HINT.format(cmd=cmd) if cmd else "")

def run_task(trainer, task, idx):
    desc = aug(task)
    bar = "=" * 70
    print("\n" + bar, flush=True)
    print("TASK %d: %s" % (idx, task.get("id", "?")), flush=True)
    print(bar, flush=True)
    print(">> 题目:", task["description"].strip(), flush=True)
    print(">> 验证命令:", task.get("test_command", ""), flush=True)
    print("", flush=True)

    print(">>> [模型生成策略]", flush=True)
    strat = trainer.generate_strategy(desc, task.get("context", "Empty project"))
    print(strat if strat else "(空)", flush=True)
    print("", flush=True)

    env = CodeGym(task, max_steps=6)
    obs = env.reset()
    print(">>> [初始环境]", obs, flush=True)
    print("", flush=True)

    actions = []
    for step in range(6):
        hist = "\n".join("Step %d: %s" % (j+1, a) for j, a in enumerate(actions[-4:]))
        action, raw, is_valid = trainer.generate_action(desc, strat, obs, hist)
        actions.append(action)
        print(">>> [第 %d 步 · 模型动作] (格式有效=%s)" % (step+1, is_valid), flush=True)
        print(action if action else "(空)", flush=True)
        print("", flush=True)
        obs, reward, done = env.step(action)
        print(">>> [沙盒反馈]", flush=True)
        print(obs, flush=True)
        print("    (reward=%.2f, done=%s)" % (reward, done), flush=True)
        print("", flush=True)
        if done:
            break

    ok = env.get_total_reward() > 0
    print(bar, flush=True)
    print(">>> 最终结果: %s  (用了 %d 步)" % ("✅ 解题成功" if ok else "❌ 未解出", len(actions)), flush=True)
    print(bar, flush=True)
    return ok

def main():
    assert os.path.isdir(MERGED), "merged model not found: " + MERGED
    print("###### 加载合并模型 (VPS 本地): " + MERGED + " ######", flush=True)
    t0 = time.time()
    cfg = StraTAConfig(model_path=MERGED, init_adapter=None,
                       max_interaction_steps=6, eval_temperature=0.7)
    trainer = StraTATrainer(cfg)
    trainer.setup()
    print("###### 模型已就绪，加载用时 %ds — 开始交互演示 ######\n" % (time.time()-t0), flush=True)

    with open(os.path.join(ROOT, "data/eval/eval_small.json")) as f:
        tasks = json.load(f)
    # 跑 3 道不同难度的题，展示真实交互
    picks = [tasks[0], tasks[3], tasks[7]]
    nsolved = 0
    for i, t in enumerate(picks):
        if run_task(trainer, t, i+1):
            nsolved += 1
    print("\n" + "#" * 70, flush=True)
    print("交互演示结束: %d/%d 题解出，模型可正常加载与多步交互。" % (nsolved, len(picks)), flush=True)

if __name__ == "__main__":
    main()

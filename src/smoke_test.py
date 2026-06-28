#!/usr/bin/env python3
"""Quick smoke test: run 2 steps of StraTA training to verify the pipeline."""
import sys
sys.path.insert(0, '/root/strata-project/src')

import json
import torch
from strata_trainer import StraTAConfig, StraTATrainer

def main():
    print("=" * 60)
    print("StraTA Quick Smoke Test (2 steps)")
    print("=" * 60)

    config = StraTAConfig(
        model_path="/root/strata-project/model",
        N=2,
        M=2,
        sigma=2,
        max_steps=2,
        batch_size=1,
        max_interaction_steps=3,
        kappa=0.0,  # disable self-judgment for smoke test
    )

    with open("/root/strata-project/data/train/synthetic.json") as f:
        all_tasks = json.load(f)
    train_tasks = all_tasks[:4]
    eval_tasks = all_tasks[:2]

    print(f"Training tasks: {len(train_tasks)}")
    print(f"Eval tasks: {len(eval_tasks)}")
    print(f"Config: N={config.N}, M={config.M}, sigma={config.sigma}")
    print()

    trainer = StraTATrainer(config)
    trainer.setup()

    print("\n>>> Starting training...")
    trainer.train(train_tasks, eval_tasks)

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()

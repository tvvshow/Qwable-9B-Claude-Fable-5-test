#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print("Step 1: imports OK")
from strata_trainer import StraTAConfig, StraTATrainer
print("Step 2: creating config...")
config = StraTAConfig(N=2, M=2, sigma=2, max_steps=2, batch_size=1, max_interaction_steps=5)
print("Step 3: creating trainer...")
trainer = StraTATrainer(config)
print("Step 4: setup (loading models)...")
trainer.setup()
print("Setup complete! GPU memory:")
import torch
print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

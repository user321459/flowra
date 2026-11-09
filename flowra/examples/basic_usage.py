"""
Minimal FLowRA example on a small LLaMA-style decoder.

    python -m flowra.examples.basic_usage
"""

import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from flowra import FLOWRA, FlowConfig, TrainingConfig  # noqa: E402
from toy_models import TinyLlama, accuracy, make_sequence_task  # noqa: E402


def main():
    torch.manual_seed(0)
    x, y = make_sequence_task("first", 2560)
    train = DataLoader(TensorDataset(x[:2048], y[:2048]), batch_size=32, shuffle=True)
    test = DataLoader(TensorDataset(x[2048:], y[2048:]), batch_size=256)
    model = TinyLlama()

    # Only two user parameters: the budget B and the composition threshold eta.
    framework = FLOWRA(model, FlowConfig(budget=0.03, eta=0.3))

    # Profiling, rank allocation and adapter injection.
    framework.adapt(train)
    print(framework.rank_table())

    # Training; ranks are refined after a 10% warm-up, then every 100 steps.
    framework.train(train, TrainingConfig(lr=0.03, max_steps=400, log_every=50),
                    eval_fn=lambda m: {"acc": accuracy(m, test)})
    print("trainable parameters:", framework.parameter_counts())

    framework.merge()  # fold the adapters into the frozen weights for inference
    print(f"test accuracy after merging: {accuracy(model, test):.4f}")


if __name__ == "__main__":
    main()

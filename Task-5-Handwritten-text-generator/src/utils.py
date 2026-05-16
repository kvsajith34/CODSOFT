"""
utils.py — Helper stuff I use everywhere.

get_device, seeding, timing, and that training plot I always end up generating.
"""

import os
import json
import math
import random
import logging
import time

import torch
import numpy as np

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Pick the fastest thing available. CUDA > MPS > CPU."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def set_seed(seed: int):
    """Make runs reproducible. I call this at the top of everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    """Turn seconds into something readable like 03:45."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def perplexity(loss: float) -> float:
    """Classic exp(loss) but I sometimes use 2**loss for bits-per-char feel."""
    return math.exp(loss)


def save_training_history(history: dict, output_dir: str):
    """Dump losses to json + optional matplotlib plot."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", path)

    # Plot training curves if matplotlib is around
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, history["train_loss"], label="Train loss")
        axes[0].plot(epochs, history["val_loss"], label="Val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, [2**l for l in history["val_loss"]], color="orange")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Perplexity (2^loss)")
        axes[1].set_title("Validation Perplexity")
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("Training curves saved to %s", plot_path)
    except ImportError:
        pass

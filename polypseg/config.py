"""Shared defaults and project paths."""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default experiment settings (match dataset_comparison.ipynb)
SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 5
SUBSET = 1000
BASE_CHANNELS = 64
LEARNING_RATE = 1e-3

# Output locations
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_CSV = PROJECT_ROOT / "model_comparison_results.csv"
LEADERBOARD_CSV = PROJECT_ROOT / "leaderboard.csv"


def get_device() -> torch.device:
    """Pick Apple GPU (mps), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

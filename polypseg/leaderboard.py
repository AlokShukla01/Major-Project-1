"""Open benchmarking framework.

Two ways to get onto the leaderboard:
  1. Submit a results CSV (same columns as model_comparison_results.csv).
  2. Submit a trained checkpoint; it is evaluated on a fixed benchmark split so every
     model is scored on exactly the same images.

The built-in benchmark is "Synthetic-Bench v1" (fixed seed procedural images), which works
without any dataset. Real benchmarks (SUN / LDPolypVideo test folders) can be added with
`evaluate_checkpoint(path, dataset=...)`.
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from .data import ArrayDataset
from .inference import load_checkpoint
from .metrics import calculate_metrics
from .synthetic import generate_arrays

REQUIRED = ["model", "dataset", "final_dice", "final_iou", "final_accuracy"]
OPTIONAL = ["best_train_loss", "best_dice", "best_iou", "samples_used"]
META = ["submitter", "source", "notes", "submitted_at"]
BENCH_NAME = "Synthetic-Bench v1"
BENCH_SEED = 2024
BENCH_SIZE = 120


def validate_submission(df: pd.DataFrame):
    """Return list of error strings (empty = valid)."""
    errors = [f"missing column '{c}'" for c in REQUIRED if c not in df.columns]
    if errors:
        return errors
    if df.empty:
        return ["submission has no rows"]
    for c in ["final_dice", "final_iou", "final_accuracy"]:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().any():
            errors.append(f"'{c}' has non-numeric values")
        elif ((vals < 0) | (vals > 1)).any():
            errors.append(f"'{c}' must be between 0 and 1")
    if df["model"].astype(str).str.strip().eq("").any():
        errors.append("empty model name")
    return errors


def load_leaderboard(path=C.LEADERBOARD_CSV, seed_results=C.RESULTS_CSV) -> pd.DataFrame:
    """Leaderboard = notebook baseline results + all submissions."""
    frames = []
    if Path(seed_results).exists():
        base = pd.read_csv(seed_results)
        base["submitter"] = "Project baseline"
        base["source"] = "notebook"
        base["notes"] = "dataset_comparison.ipynb"
        base["submitted_at"] = ""
        frames.append(base)
    if Path(path).exists():
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=REQUIRED + OPTIONAL + META)
    df = pd.concat(frames, ignore_index=True)
    for c in ["final_dice", "final_iou", "final_accuracy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_submission(df: pd.DataFrame, submitter: str, source="csv", notes="", path=C.LEADERBOARD_CSV):
    errors = validate_submission(df)
    if errors:
        raise ValueError("; ".join(errors))
    keep = [c for c in REQUIRED + OPTIONAL if c in df.columns]
    sub = df[keep].copy()
    sub["submitter"] = submitter.strip() or "anonymous"
    sub["source"] = source
    sub["notes"] = notes
    sub["submitted_at"] = dt.datetime.now().isoformat(timespec="seconds")
    path = Path(path)
    if path.exists():
        sub = pd.concat([pd.read_csv(path), sub], ignore_index=True)
    sub.to_csv(path, index=False)
    return len(df)


def rank(df: pd.DataFrame, metric="final_dice", dataset=None) -> pd.DataFrame:
    view = df if dataset in (None, "All") else df[df["dataset"] == dataset]
    view = view.sort_values(metric, ascending=False).reset_index(drop=True)
    view.insert(0, "rank", range(1, len(view) + 1))
    return view


def overall_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of Dice and IoU over all datasets a model was evaluated on."""
    g = df.groupby(["model", "submitter"], as_index=False).agg(
        datasets=("dataset", "nunique"),
        mean_dice=("final_dice", "mean"),
        mean_iou=("final_iou", "mean"),
        mean_accuracy=("final_accuracy", "mean"),
    )
    g["score"] = (g["mean_dice"] + g["mean_iou"]) / 2
    return g.sort_values("score", ascending=False).reset_index(drop=True)


_BENCH_CACHE = {}


def benchmark_dataset(image_size=C.IMAGE_SIZE):
    key = tuple(image_size)
    if key not in _BENCH_CACHE:
        imgs, masks, _, _ = generate_arrays(BENCH_SIZE, seed=BENCH_SEED)
        _BENCH_CACHE[key] = ArrayDataset(imgs, masks, key)
    return _BENCH_CACHE[key]


def evaluate_checkpoint(path, dataset=None, dataset_name=BENCH_NAME, batch_size=8, device=None):
    """Evaluate a checkpoint on a fixed benchmark. Returns a leaderboard row dict."""
    model, meta = load_checkpoint(path, device)
    dataset = dataset or benchmark_dataset(meta["image_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dev = next(model.parameters()).device
    acc = dice = iou = 0.0
    with torch.no_grad():
        for x, y in loader:
            a, d, i = calculate_metrics(model(x.to(dev)), y.to(dev))
            acc, dice, iou = acc + a, dice + d, iou + i
    n = len(loader)
    label = meta["model_name"] + ("" if meta.get("variant", "baseline") == "baseline" else f" ({meta['variant']})")
    return {
        "model": label,
        "dataset": dataset_name,
        "final_dice": dice / n,
        "final_iou": iou / n,
        "final_accuracy": acc / n,
        "samples_used": len(dataset),
        "trained_on": meta.get("dataset"),
    }

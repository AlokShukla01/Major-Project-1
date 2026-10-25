"""Automated training pipeline.

Python API
----------
    from polypseg.train import TrainConfig, run_experiment
    result = run_experiment(dataset, "SUN Database", "DeepLabV3+", TrainConfig(epochs=5))

CLI
---
    # Smoke test on procedural synthetic data (no real dataset needed)
    python -m polypseg.train --data synthetic --models "Basic U-Net" --epochs 2 --base 16

    # Real data (same folder layouts as the notebook)
    python -m polypseg.train --data "sun:/path/to/sundatabase_positive_part1" --models all
    python -m polypseg.train --data "ldpolyp:/path/to/LD Polyp/TrainValid" --models all --curriculum

    # Synthetic augmentation experiment: add 500 synthetic images to the training split
    python -m polypseg.train --data "ldpolyp:/path" --models "U-Net++" --mix-synthetic 500

Every run saves a checkpoint to checkpoints/, logs to TensorBoard under runs/ and appends a
row (same columns as model_comparison_results.csv) to the results CSV.
"""

import argparse
import csv
import datetime as dt
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset

from . import config as C
from .curriculum import curriculum_indices, score_dataset
from .data import ArrayDataset, LDPolypDataset, SUNPolypDataset
from .metrics import calculate_metrics
from .models import MODEL_FACTORIES, build_model, count_parameters
from .synthetic import generate_arrays

RESULT_COLUMNS = [
    "model", "dataset", "best_train_loss", "best_val_accuracy", "best_dice", "best_iou",
    "final_train_loss", "final_accuracy", "final_dice", "final_iou", "samples_used",
]
PIPELINE_RESULTS_CSV = C.OUTPUT_DIR / "pipeline_results.csv"


@dataclass
class TrainConfig:
    epochs: int = C.EPOCHS
    batch_size: int = C.BATCH_SIZE
    lr: float = C.LEARNING_RATE
    subset: int = C.SUBSET
    base: int = C.BASE_CHANNELS
    image_size: tuple = C.IMAGE_SIZE
    seed: int = C.SEED
    val_fraction: float = 0.2
    curriculum: bool = False
    curriculum_start: float = 0.3
    mix_synthetic: int = 0
    device: str = field(default_factory=lambda: str(C.get_device()))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def slug(text: str) -> str:
    text = text.lower().replace("+", "plus")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _make_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # tensorboard not installed
        return None
    return SummaryWriter(log_dir=str(log_dir))


def split_dataset(dataset, cfg: TrainConfig):
    """Subset + reproducible train/val split, same logic as the notebook."""
    sample_count = min(cfg.subset, len(dataset))
    if sample_count < 2:
        raise ValueError(f"Need at least 2 samples, found {sample_count}.")
    sub = Subset(dataset, list(range(sample_count)))
    train_n = max(1, int((1 - cfg.val_fraction) * len(sub)))
    val_n = len(sub) - train_n
    if val_n == 0:
        train_n, val_n = train_n - 1, 1
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = torch.utils.data.random_split(sub, [train_n, val_n], generator=gen)
    return train_ds, val_ds, sample_count


def _raw_indices(train_ds):
    """Map a random_split Subset of a Subset back to indices of the base dataset."""
    outer = train_ds.indices
    inner = train_ds.dataset.indices
    return [inner[i] for i in outer], train_ds.dataset.dataset


def evaluate(model, loader, device):
    model.eval()
    acc = dice = iou = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            a, d, i = calculate_metrics(model(imgs), masks)
            acc, dice, iou = acc + a, dice + d, iou + i
    n = max(1, len(loader))
    return acc / n, dice / n, iou / n


def run_experiment(dataset, dataset_name, model_name, cfg: TrainConfig = None,
                   progress_callback=None, save_checkpoint=True, results_csv=PIPELINE_RESULTS_CSV,
                   run_tag=None):
    """Train one model on one dataset and return a result dict.

    progress_callback(epoch, total_epochs, metrics_dict) is called after every epoch
    (the dashboard uses it to draw live charts).
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_ds, val_ds, sample_count = split_dataset(dataset, cfg)

    if cfg.mix_synthetic > 0:
        s_imgs, s_masks, _, _ = generate_arrays(cfg.mix_synthetic, seed=cfg.seed + 1)
        train_ds = ConcatDataset([train_ds, ArrayDataset(s_imgs, s_masks, cfg.image_size)])

    scores = None
    if cfg.curriculum:
        if cfg.mix_synthetic > 0:
            raise ValueError("Use either --curriculum or --mix-synthetic, not both.")
        raw_idx, base_ds = _raw_indices(train_ds)
        scores = score_dataset(base_ds, raw_idx)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(cfg.seed)
    model = build_model(model_name, base=cfg.base).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    variant = run_tag or ("curriculum" if cfg.curriculum else (f"synth{cfg.mix_synthetic}" if cfg.mix_synthetic else "baseline"))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = _make_writer(C.RUNS_DIR / f"{slug(model_name)}__{slug(dataset_name)}__{variant}__{stamp}")

    history = {"loss": [], "acc": [], "dice": [], "iou": [], "train_samples": [], "seconds": []}
    for epoch in range(cfg.epochs):
        t0 = time.time()
        if scores is not None:
            pos = curriculum_indices(scores, epoch, cfg.epochs, start_fraction=cfg.curriculum_start)
            epoch_ds = Subset(train_ds, pos.tolist())
        else:
            epoch_ds = train_ds
        train_loader = DataLoader(epoch_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        acc, dice, iou = evaluate(model, val_loader, device)
        secs = time.time() - t0
        for k, v in zip(["loss", "acc", "dice", "iou", "train_samples", "seconds"],
                        [avg_loss, acc, dice, iou, len(epoch_ds), secs]):
            history[k].append(v)

        if writer:
            writer.add_scalar("train/loss", avg_loss, epoch + 1)
            writer.add_scalar("val/accuracy", acc, epoch + 1)
            writer.add_scalar("val/dice", dice, epoch + 1)
            writer.add_scalar("val/iou", iou, epoch + 1)
            writer.add_scalar("train/samples", len(epoch_ds), epoch + 1)

        msg = {"loss": avg_loss, "acc": acc, "dice": dice, "iou": iou, "train_samples": len(epoch_ds), "seconds": secs}
        print(f"  [{model_name} | {dataset_name} | {variant}] epoch {epoch + 1}/{cfg.epochs} "
              f"loss {avg_loss:.4f} acc {acc:.4f} dice {dice:.4f} iou {iou:.4f} "
              f"({len(epoch_ds)} train samples, {secs:.1f}s)")
        if progress_callback:
            progress_callback(epoch + 1, cfg.epochs, msg)

    if writer:
        writer.add_hparams(
            {"model": model_name, "dataset": dataset_name, "variant": variant,
             "epochs": cfg.epochs, "base": cfg.base, "lr": cfg.lr},
            {"hparam/final_dice": history["dice"][-1], "hparam/final_iou": history["iou"][-1]},
        )
        writer.close()

    row = {
        "model": model_name if variant == "baseline" else f"{model_name} ({variant})",
        "dataset": dataset_name,
        "best_train_loss": min(history["loss"]),
        "best_val_accuracy": max(history["acc"]),
        "best_dice": max(history["dice"]),
        "best_iou": max(history["iou"]),
        "final_train_loss": history["loss"][-1],
        "final_accuracy": history["acc"][-1],
        "final_dice": history["dice"][-1],
        "final_iou": history["iou"][-1],
        "samples_used": sample_count,
    }

    ckpt_path = None
    if save_checkpoint:
        C.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = C.CHECKPOINT_DIR / f"{slug(model_name)}__{slug(dataset_name)}__{variant}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "base": cfg.base,
            "image_size": list(cfg.image_size),
            "dataset": dataset_name,
            "variant": variant,
            "history": history,
            "metrics": row,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
            "parameters": count_parameters(model),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
        }, ckpt_path)

    if results_csv:
        append_result(row, results_csv)

    return {"row": row, "history": history, "checkpoint": ckpt_path, "model": model}


def append_result(row: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: row[k] for k in RESULT_COLUMNS})


def parse_data_arg(spec: str, cfg: TrainConfig, synthetic_n=300):
    """'synthetic' | 'sun:<path>' | 'ldpolyp:<path>' -> (dataset, display_name)."""
    if spec == "synthetic":
        imgs, masks, _, _ = generate_arrays(synthetic_n, seed=cfg.seed)
        return ArrayDataset(imgs, masks, cfg.image_size), "Synthetic"
    kind, _, path = spec.partition(":")
    if kind == "sun":
        return SUNPolypDataset(path, cfg.image_size), "SUN Database"
    if kind == "ldpolyp":
        return LDPolypDataset(path, cfg.image_size), "LDPolypVideo"
    raise ValueError(f"Unrecognised --data value: {spec}")


def main(argv=None):
    p = argparse.ArgumentParser(description="PolypSeg automated training pipeline")
    p.add_argument("--data", required=True, help="synthetic | sun:<path> | ldpolyp:<path>")
    p.add_argument("--models", default="all", help='"all" or comma-separated model names')
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    p.add_argument("--subset", type=int, default=C.SUBSET)
    p.add_argument("--base", type=int, default=C.BASE_CHANNELS, help="base channel width")
    p.add_argument("--curriculum", action="store_true", help="train with curriculum learning")
    p.add_argument("--compare-curriculum", action="store_true",
                   help="run baseline AND curriculum for each model")
    p.add_argument("--mix-synthetic", type=int, default=0, help="extra synthetic training images")
    p.add_argument("--synthetic-n", type=int, default=300, help="size of --data synthetic")
    p.add_argument("--results-csv", default=str(PIPELINE_RESULTS_CSV))
    p.add_argument("--device", default=str(C.get_device()))
    args = p.parse_args(argv)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, subset=args.subset,
                      base=args.base, curriculum=args.curriculum, mix_synthetic=args.mix_synthetic,
                      device=args.device)
    dataset, ds_name = parse_data_arg(args.data, cfg, args.synthetic_n)
    names = list(MODEL_FACTORIES) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    print(f"Dataset {ds_name}: {len(dataset)} samples | device {cfg.device} | models {names}")

    for name in names:
        if args.compare_curriculum:
            for use_cur in (False, True):
                cfg.curriculum = use_cur
                run_experiment(dataset, ds_name, name, cfg, results_csv=args.results_csv)
        else:
            run_experiment(dataset, ds_name, name, cfg, results_csv=args.results_csv)
    print(f"Results appended to {args.results_csv}")


if __name__ == "__main__":
    main()

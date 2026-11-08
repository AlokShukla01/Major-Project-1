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
import copy
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
    augment: bool = False
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
    """Reproducible subset and split, keeping video sequences/cases together when possible."""
    sample_count = min(cfg.subset, len(dataset))
    if sample_count < 2:
        raise ValueError(f"Need at least 2 samples, found {sample_count}.")
    # Select across the whole dataset before splitting. Taking the first N frames
    # clustered the LDPolyp sample in its first sorted sequences and biased runs.
    selection_gen = torch.Generator().manual_seed(cfg.seed)
    selected = torch.randperm(len(dataset), generator=selection_gen)[:sample_count].tolist()
    sub = Subset(dataset, selected)
    gen = torch.Generator().manual_seed(cfg.seed)
    val_n = max(1, int(cfg.val_fraction * len(sub)))

    # Frames from one LDPolyp sequence (or SUN case) are highly correlated. Keep
    # groups on one side of validation to avoid near-duplicate frame leakage.
    grouped = hasattr(dataset, "samples") and len(dataset.samples) == len(dataset)
    if grouped:
        group_indices = {}
        for local_idx, source_idx in enumerate(selected):
            path = dataset.samples[source_idx][0]
            group_indices.setdefault(Path(path).parent.name, []).append(local_idx)
        groups = list(group_indices)
        if len(groups) > 1:
            order = torch.randperm(len(groups), generator=gen).tolist()
            val_groups, count = [], 0
            for group_idx in order:
                if count >= val_n or len(val_groups) >= len(groups) - 1:
                    break
                group = groups[group_idx]
                val_groups.append(group)
                count += len(group_indices[group])
            val_set = set(val_groups)
            train_local = [i for g, idxs in group_indices.items() if g not in val_set for i in idxs]
            val_local = [i for g, idxs in group_indices.items() if g in val_set for i in idxs]
            train_ds, val_ds = Subset(sub, train_local), Subset(sub, val_local)
        else:
            train_n = len(sub) - val_n
            train_ds, val_ds = torch.utils.data.random_split(sub, [train_n, val_n], generator=gen)
    else:
        train_n = len(sub) - val_n
        train_ds, val_ds = torch.utils.data.random_split(sub, [train_n, val_n], generator=gen)
    return train_ds, val_ds, sample_count


def _raw_indices(train_ds):
    """Map a random_split Subset of a Subset back to indices of the base dataset."""
    outer = train_ds.indices
    inner = train_ds.dataset.indices
    return [inner[i] for i in outer], train_ds.dataset.dataset


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    acc = dice = iou = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            a, d, i = calculate_metrics(model(imgs), masks, threshold)
            acc, dice, iou = acc + a, dice + d, iou + i
    n = max(1, len(loader))
    return acc / n, dice / n, iou / n


def segmentation_loss(logits, masks):
    """Class-balanced BCE plus soft Dice for sparse polyp regions.

    A plain BCE objective can minimize loss by predicting background everywhere
    when target boxes cover few pixels. Batchwise positive weighting and Dice
    overlap keep foreground pixels represented in the learning signal.
    """
    positive = masks.sum()
    negative = masks.numel() - positive
    pos_weight = (negative / positive.clamp_min(1)).clamp(min=1.0, max=20.0)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, masks, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, masks.ndim))
    intersection = (probs * masks).sum(dim=dims)
    denominator = probs.sum(dim=dims) + masks.sum(dim=dims)
    dice_loss = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return bce + dice_loss


def augment_batch(images, masks):
    """Apply paired flips and mild color variation to colonoscopy training batches."""
    if random.random() < 0.5:
        images, masks = images.flip(-1), masks.flip(-1)
    if random.random() < 0.25:
        images, masks = images.flip(-2), masks.flip(-2)
    contrast = random.uniform(0.9, 1.1)
    brightness = random.uniform(0.9, 1.1)
    images = (((images - 0.5) * contrast + 0.5) * brightness).clamp_(0, 1)
    return images, masks


def run_experiment(dataset, dataset_name, model_name, cfg: TrainConfig = None,
                   progress_callback=None, save_checkpoint=True, results_csv=PIPELINE_RESULTS_CSV,
                   run_tag=None, initial_checkpoint=None):
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
    validation_threshold = 0.5
    if initial_checkpoint:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if "state_dict" not in initial:
            raise ValueError(f"Initial checkpoint has no state_dict: {initial_checkpoint}")
        model.load_state_dict(initial["state_dict"])
        validation_threshold = float(initial.get("video_threshold", initial.get("threshold", 0.5)))
        print(f"Initialized {model_name} from {initial_checkpoint}")
    if model_name == "Pretrained ResNet18 U-Net":
        if not model.pretrained_encoder:
            print("WARNING: ImageNet ResNet-18 weights are not cached; training encoder from scratch.")
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        else:
            optimizer = optim.Adam([
                {"params": model.encoder.parameters(), "lr": cfg.lr * 0.1},
                {"params": list(model.decoder3.parameters()) + list(model.decoder2.parameters()) +
                           list(model.decoder1.parameters()) + list(model.decoder0.parameters()) +
                           list(model.head.parameters()), "lr": cfg.lr},
            ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    variant = run_tag or ("curriculum" if cfg.curriculum else (f"synth{cfg.mix_synthetic}" if cfg.mix_synthetic else "baseline"))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = _make_writer(C.RUNS_DIR / f"{slug(model_name)}__{slug(dataset_name)}__{variant}__{stamp}")

    history = {"loss": [], "acc": [], "dice": [], "iou": [], "train_samples": [], "seconds": []}
    best_state = None
    best_epoch = 0
    best_validation = None
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
        for batch_index, (imgs, masks) in enumerate(train_loader, start=1):
            if cfg.augment:
                imgs, masks = augment_batch(imgs, masks)
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = segmentation_loss(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if model_name == "Polyp-PVT" and batch_index % 100 == 0:
                print(f"    epoch {epoch + 1}: batch {batch_index}/{len(train_loader)} "
                      f"loss {loss.item():.4f}", flush=True)

        avg_loss = total_loss / max(1, len(train_loader))
        acc, dice, iou = evaluate(model, val_loader, device, threshold=validation_threshold)
        if best_validation is None or dice > best_validation["dice"]:
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_validation = {"loss": avg_loss, "acc": acc, "dice": dice, "iou": iou}
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

    model.load_state_dict(best_state)

    row = {
        "model": model_name if variant == "baseline" else f"{model_name} ({variant})",
        "dataset": dataset_name,
        "best_train_loss": min(history["loss"]),
        "best_val_accuracy": max(history["acc"]),
        "best_dice": max(history["dice"]),
        "best_iou": max(history["iou"]),
        "final_train_loss": best_validation["loss"],
        "final_accuracy": best_validation["acc"],
        "final_dice": best_validation["dice"],
        "final_iou": best_validation["iou"],
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
            "selected_epoch": best_epoch,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
            "loss": "class_balanced_bce_plus_soft_dice",
            "pretrained_encoder": getattr(model, "pretrained_encoder", False),
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
            "video_threshold": validation_threshold,
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
    p.add_argument("--image-size", type=int, choices=[128, 256, 384], default=C.IMAGE_SIZE[0],
                   help="square input resolution")
    p.add_argument("--augment", action="store_true", help="paired flips and mild color augmentation")
    p.add_argument("--curriculum", action="store_true", help="train with curriculum learning")
    p.add_argument("--compare-curriculum", action="store_true",
                   help="run baseline AND curriculum for each model")
    p.add_argument("--mix-synthetic", type=int, default=0, help="extra synthetic training images")
    p.add_argument("--run-tag", help="optional checkpoint/result variant name")
    p.add_argument("--init-checkpoint", help="initialize model weights from a compatible checkpoint")
    p.add_argument("--synthetic-n", type=int, default=300, help="size of --data synthetic")
    p.add_argument("--results-csv", default=str(PIPELINE_RESULTS_CSV))
    p.add_argument("--device", default=str(C.get_device()))
    args = p.parse_args(argv)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, subset=args.subset,
                      base=args.base, curriculum=args.curriculum, mix_synthetic=args.mix_synthetic,
                      image_size=(args.image_size, args.image_size), augment=args.augment,
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
            run_experiment(dataset, ds_name, name, cfg, results_csv=args.results_csv,
                           run_tag=args.run_tag, initial_checkpoint=args.init_checkpoint)
    print(f"Results appended to {args.results_csv}")


if __name__ == "__main__":
    main()

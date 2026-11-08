"""Evaluate local LDPolypVideo checkpoints on the held-out Test split."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import Subset

from . import config as C
from .data import LDPolypDataset
from .inference import list_checkpoints, load_checkpoint
from .metrics import calculate_metrics
from .temporal import VideoClipDataset
from torch.utils.data import DataLoader


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained LDPolypVideo checkpoints on Test")
    parser.add_argument("--data", default=str(C.PROJECT_ROOT / "LD Polyp" / "Test"),
                        help="LDPolypVideo Test split root")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="fixed-size test subset; use 0 to evaluate the full split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", help="evaluate only this model name")
    parser.add_argument("--variant", help="evaluate only this checkpoint variant, e.g. video256")
    parser.add_argument("--checkpoint", help="evaluate this checkpoint regardless of its training dataset")
    parser.add_argument("--threshold", type=float,
                        help="probability threshold (default: checkpoint calibration, else 0.5)")
    parser.add_argument("--output", default=str(C.OUTPUT_DIR / "ldpolyp_test_results.csv"))
    args = parser.parse_args(argv)

    root = Path(args.data).expanduser()
    index_dataset = LDPolypDataset(root)
    if len(index_dataset) < 1:
        parser.error(f"No annotated test images found under {args.data}")
    indices = None
    if args.max_samples > 0 and args.max_samples < len(index_dataset):
        gen = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(index_dataset), generator=gen)[:args.max_samples].tolist()

    checkpoints = [c for c in list_checkpoints()
                   if ((args.checkpoint and Path(args.checkpoint).expanduser().resolve() == c["path"].resolve())
                       or (not args.checkpoint and c.get("dataset") == "LDPolypVideo"))
                   and (args.model is None or c.get("model_name") == args.model)
                   and (args.variant is None or c.get("variant") == args.variant)]
    if not checkpoints:
        parser.error("No LDPolypVideo checkpoints found in checkpoints/")

    rows = []
    for ck in checkpoints:
        frames = LDPolypDataset(root, image_size=ck.get("image_size", C.IMAGE_SIZE))
        model, meta = load_checkpoint(ck["path"])
        if meta["model_name"] == "Temporal Attention U-Net":
            dataset = VideoClipDataset(frames, image_size=meta.get("image_size", C.IMAGE_SIZE),
                                       window=meta.get("temporal_window", 5))
        else:
            dataset = frames
        if indices is not None:
            dataset = Subset(dataset, indices)
        threshold = args.threshold if args.threshold is not None else meta.get(
            "video_threshold", meta.get("threshold", 0.5))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        device = next(model.parameters()).device
        totals = [0.0, 0.0, 0.0]
        with torch.no_grad():
            for images, masks in loader:
                values = calculate_metrics(model(images.to(device)), masks.to(device), threshold)
                totals = [a + b for a, b in zip(totals, values)]
        row = {"model": meta["model_name"], "dataset": "LDPolypVideo Test",
               "final_accuracy": totals[0] / len(loader), "final_dice": totals[1] / len(loader),
               "final_iou": totals[2] / len(loader), "threshold": threshold, "samples_used": len(dataset),
               "trained_on": meta.get("dataset"), "variant": ck["variant"]}
        row["checkpoint"] = ck["path"].name
        rows.append(row)
        print(f"{row['model']} ({row['variant']}): Dice {row['final_dice']:.4f} | "
              f"IoU {row['final_iou']:.4f} | accuracy {row['final_accuracy']:.4f}")

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: row["final_dice"], reverse=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluated {len(indices) if indices is not None else len(index_dataset):,} shared test images; results saved to {out}")


if __name__ == "__main__":
    main()

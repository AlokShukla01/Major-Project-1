"""Choose a video mask threshold on held-out TrainValid sequences."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import LDPolypDataset
from .inference import load_checkpoint
from .train import TrainConfig, split_dataset


@torch.no_grad()
def best_dice_threshold(model, loader, device):
    thresholds = [i / 100 for i in range(10, 91, 5)]
    totals = {t: [0, 0] for t in thresholds}
    model.eval()
    for images, masks in loader:
        probs = torch.sigmoid(model(images.to(device)))
        masks = masks.to(device).bool()
        for threshold in thresholds:
            pred = probs >= threshold
            intersection = (pred & masks).sum().item()
            denom = pred.sum().item() + masks.sum().item()
            totals[threshold][0] += 2 * intersection
            totals[threshold][1] += denom
    scores = {t: (num + 1e-6) / (den + 1e-6) for t, (num, den) in totals.items()}
    threshold = max(thresholds, key=lambda t: scores[t])
    return threshold, scores[threshold]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate a checkpoint's video probability threshold")
    parser.add_argument("checkpoint", help="checkpoint produced by polypseg.train")
    parser.add_argument("--data", default="LD Polyp/TrainValid", help="LDPolypVideo TrainValid root")
    args = parser.parse_args(argv)

    path = Path(args.checkpoint).expanduser()
    meta = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(meta.get("config", {}))
    cfg["image_size"] = tuple(cfg.get("image_size", meta.get("image_size", (128, 128))))
    config = TrainConfig(**{k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__})
    dataset = LDPolypDataset(args.data, image_size=config.image_size, skip_empty=False)
    _, val_set, count = split_dataset(dataset, config)
    loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model, _ = load_checkpoint(path, device=config.device)
    threshold, dice = best_dice_threshold(model, loader, config.device)

    meta["video_threshold"] = threshold
    meta["threshold_calibration"] = {
        "split": "held-out LDPolypVideo TrainValid sequences",
        "samples_used": count,
        "validation_samples": len(val_set),
        "dice": dice,
        "thresholds": [i / 100 for i in range(10, 91, 5)],
    }
    torch.save(meta, path)
    print(f"Calibrated threshold {threshold:.2f} (validation Dice {dice:.4f}, "
          f"{len(val_set):,} held-out frames); updated {path}")


if __name__ == "__main__":
    main()

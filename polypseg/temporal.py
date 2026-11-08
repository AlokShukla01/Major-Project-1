"""Video clip dataset and training entry point for temporal segmentation."""

import argparse
import csv
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from . import config as C
from .data import LDPolypDataset
from .metrics import calculate_metrics
from .models import TemporalAttentionUNet, count_parameters
from .train import augment_batch, segmentation_loss


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", Path(path).stem)]


class VideoClipDataset(Dataset):
    """Return a short same-video clip and the centre frame's rectangular label."""

    def __init__(self, frames, image_size=(128, 128), window=5, clip_stride=1):
        if window < 3 or window % 2 != 1:
            raise ValueError("window must be an odd number of at least 3")
        self.frames = frames
        self.image_size = image_size
        self.window = window
        self.clip_stride = max(1, int(clip_stride))
        grouped = {}
        for idx, (path, _) in enumerate(frames.samples):
            grouped.setdefault(Path(path).parent, []).append(idx)
        self.sequences = {
            seq: sorted(indices, key=lambda i: _natural_key(frames.samples[i][0]))
            for seq, indices in grouped.items()
        }
        self.sequence_names = list(self.sequences)
        self.clips = []
        radius = window // 2
        for seq, indices in self.sequences.items():
            for pos in range(0, len(indices), self.clip_stride):
                center_idx = indices[pos]
                neighbors = [indices[min(max(pos + offset, 0), len(indices) - 1)]
                             for offset in range(-radius, radius + 1)]
                self.clips.append((seq, neighbors, center_idx))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        _, indices, center_idx = self.clips[index]
        images, masks = zip(*(self.frames[i] for i in indices))
        clip = torch.stack(images, dim=0)
        # All transforms are applied consistently to a clip.
        return clip, masks[len(masks) // 2]


def _split_by_sequence(dataset, val_fraction=0.2, seed=42):
    names = dataset.sequence_names
    if len(names) < 2:
        raise ValueError("At least two video sequences are needed for a grouped validation split.")
    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed)).tolist()
    val_count = max(1, min(len(names) - 1, round(len(names) * val_fraction)))
    val_names = {names[i] for i in order[:val_count]}
    train_idx, val_idx = [], []
    for i, (seq, _, _) in enumerate(dataset.clips):
        (val_idx if seq in val_names else train_idx).append(i)
    return Subset(dataset, train_idx), Subset(dataset, val_idx), sorted(str(x) for x in val_names)


def _evaluate(model, loader, device, threshold=0.5):
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64)
    with torch.no_grad():
        for clips, masks in loader:
            logits = model(clips.to(device))
            pred = (torch.sigmoid(logits) > threshold)
            target = masks.to(device).bool()
            totals[0] += (pred & target).sum().item()
            totals[1] += pred.sum().item()
            totals[2] += target.sum().item()
            totals[3] += (pred == target).sum().item()
    inter, pred_n, target_n, correct = totals.tolist()
    dice = (2 * inter + 1e-6) / (pred_n + target_n + 1e-6)
    iou = (inter + 1e-6) / (pred_n + target_n - inter + 1e-6)
    accuracy = correct / max(1, len(loader.dataset) * masks.shape[-1] * masks.shape[-2])
    return accuracy, dice, iou


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train temporal attention U-Net on LDPolypVideo clips")
    parser.add_argument("--data", default=str(C.PROJECT_ROOT / "LD Polyp" / "TrainValid"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--clip-stride", type=int, default=2,
                        help="Use every Nth frame as a centre frame, while keeping adjacent context frames.")
    parser.add_argument("--device", default=str(C.get_device()))
    parser.add_argument("--run-tag", default="temporal-attention")
    parser.add_argument("--output", default=str(C.CHECKPOINT_DIR / "temporal-attention-unet__ldpolypvideo.pt"))
    parser.add_argument("--results-csv", default=str(C.OUTPUT_DIR / "temporal_training.csv"))
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    image_size = (args.image_size, args.image_size)
    frames = LDPolypDataset(args.data, image_size=image_size)
    clips = VideoClipDataset(frames, image_size=image_size, window=args.window,
                             clip_stride=args.clip_stride)
    train_ds, val_ds, val_sequences = _split_by_sequence(clips)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TemporalAttentionUNet(base=args.base, temporal_window=args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best = -1.0
    best_state = None
    history = []
    print(f"Temporal Attention U-Net: {len(clips):,} clips across {len(clips.sequence_names)} videos; "
          f"train={len(train_ds):,}, validation={len(val_ds):,} | device={device}")
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        losses = 0.0
        for clips_x, masks in train_loader:
            if torch.rand(()) < 0.5:
                clips_x, masks = clips_x.flip(-1), masks.flip(-1)
            if torch.rand(()) < 0.25:
                clips_x, masks = clips_x.flip(-2), masks.flip(-2)
            contrast = torch.empty(1).uniform_(0.9, 1.1).item()
            brightness = torch.empty(1).uniform_(0.9, 1.1).item()
            clips_x = (((clips_x - 0.5) * contrast + 0.5) * brightness).clamp_(0, 1)
            clips_x, masks = clips_x.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(clips_x), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses += loss.item()
        scheduler.step()
        val = _evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, "loss": losses / max(1, len(train_loader)),
                  "accuracy": val[0], "dice": val[1], "iou": val[2],
                  "seconds": time.time() - started}
        history.append(record)
        print(f"epoch {epoch + 1}/{args.epochs} loss={record['loss']:.4f} "
              f"val_dice={val[1]:.4f} val_iou={val[2]:.4f} ({record['seconds']:.1f}s)", flush=True)
        if val[1] > best:
            best = val[1]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    ckpt_path = Path(args.output)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": "Temporal Attention U-Net",
                "base": args.base, "image_size": list(image_size), "dataset": "LDPolypVideo",
                "variant": args.run_tag, "temporal_window": args.window, "threshold": 0.5,
                "history": history, "metrics": history[max(range(len(history)), key=lambda i: history[i]["dice"])],
                "validation_sequences": val_sequences, "parameters": count_parameters(model)}, ckpt_path)
    csv_path = Path(args.results_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Best validation Dice: {best:.4f}; checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()

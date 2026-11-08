"""Wrap the official Polyp-PVT weights in a safe PolypSeg checkpoint."""

import argparse
from pathlib import Path

import torch

from .models import build_model, count_parameters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="checkpoints/PolypPVT_official.pth")
    parser.add_argument("--output", default="checkpoints/polyp-pvt__official-zero-shot.pt")
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args(argv)

    model = build_model("Polyp-PVT", base=32)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state, strict=True)
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": "Polyp-PVT",
        "base": 32,
        "image_size": [args.image_size, args.image_size],
        "dataset": "External polyp-specific pretrained",
        "trained_on": "Official Polyp-PVT weights",
        "variant": "official-zero-shot",
        "threshold": 0.5,
        "config": {
            "epochs": 5, "batch_size": 4, "lr": 1e-4, "subset": 1000000,
            "base": 32, "image_size": [args.image_size, args.image_size], "seed": 42,
            "val_fraction": 0.2, "curriculum": False, "curriculum_start": 0.3,
            "mix_synthetic": 0, "augment": False, "device": "mps",
        },
        "parameters": count_parameters(model),
        "source": "https://github.com/DengPingFan/Polyp-PVT",
    }, output)
    print(f"Imported official Polyp-PVT weights ({count_parameters(model):,} parameters) -> {output}")


if __name__ == "__main__":
    main()

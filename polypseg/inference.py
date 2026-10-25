"""Checkpoint discovery, loading and prediction."""

from pathlib import Path

import cv2
import numpy as np
import torch

from . import config as C
from .models import build_model


def list_checkpoints(directory=C.CHECKPOINT_DIR):
    """Return metadata dicts for every *.pt checkpoint (newest first)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            meta = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as exc:  # corrupted / foreign file
            out.append({"path": p, "error": str(exc)})
            continue
        out.append({
            "path": p,
            "label": f"{meta.get('model_name')} | {meta.get('dataset')} | {meta.get('variant', 'baseline')}",
            "model_name": meta.get("model_name"),
            "dataset": meta.get("dataset"),
            "variant": meta.get("variant", "baseline"),
            "metrics": meta.get("metrics", {}),
            "history": meta.get("history", {}),
            "created": meta.get("created"),
            "parameters": meta.get("parameters"),
        })
    return [m for m in out if "error" not in m]


def load_checkpoint(path, device=None):
    """Load a checkpoint saved by polypseg.train. Returns (model in eval mode, metadata).

    weights_only=True: refuses arbitrary pickled objects, so uploaded checkpoints
    (e.g. leaderboard submissions) cannot execute code on load.
    """
    device = torch.device(device) if device else C.get_device()
    meta = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(meta, dict) or "state_dict" not in meta or "model_name" not in meta:
        raise ValueError("Not a PolypSeg checkpoint (expected keys: state_dict, model_name).")
    model = build_model(meta["model_name"], base=meta.get("base", C.BASE_CHANNELS))
    model.load_state_dict(meta["state_dict"])
    model.to(device).eval()
    meta["image_size"] = tuple(meta.get("image_size", C.IMAGE_SIZE))
    return model, meta


def preprocess(img_rgb: np.ndarray, image_size=C.IMAGE_SIZE) -> torch.Tensor:
    x = cv2.resize(img_rgb, tuple(image_size)).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)


@torch.no_grad()
def predict_prob(model, img_rgb: np.ndarray, image_size=C.IMAGE_SIZE, device=None) -> np.ndarray:
    """Polyp probability map (float32 in [0,1]) at the original image resolution."""
    device = device or next(model.parameters()).device
    x = preprocess(img_rgb, image_size).to(device)
    prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    h, w = img_rgb.shape[:2]
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def heuristic_prob(img_rgb: np.ndarray) -> np.ndarray:
    """Untrained colour heuristic (red minus half green). Only a UI fallback, not a model."""
    arr = img_rgb.astype(np.float32) / 255.0
    score = arr[..., 0] - 0.5 * arr[..., 1]
    return (score - score.min()) / (score.max() - score.min() + 1e-8)


def mask_to_boxes(mask: np.ndarray, prob: np.ndarray = None, min_area: int = 50):
    """Connected components -> list of dicts {box:(x1,y1,x2,y2), area, score}."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        score = float(prob[labels == i].mean()) if prob is not None else 1.0
        boxes.append({"box": (int(x), int(y), int(x + w), int(y + h)), "area": int(area), "score": score})
    return sorted(boxes, key=lambda b: b["score"], reverse=True)


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, alpha=0.45, colour=(0, 255, 0)) -> np.ndarray:
    base = img_rgb.astype(np.float32)
    tint = np.zeros_like(base)
    tint[...] = colour
    m3 = mask[..., None].astype(np.float32)
    return (base * (1 - alpha * m3) + tint * (alpha * m3)).astype(np.uint8)


def draw_boxes(img_rgb: np.ndarray, boxes, colour=(0, 255, 0), label="polyp") -> np.ndarray:
    out = img_rgb.copy()
    thick = max(1, round(min(out.shape[:2]) / 200))
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick + 1)
        text = f"{label} {b['score']:.2f}"
        scale = max(0.4, min(out.shape[:2]) / 700)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return out

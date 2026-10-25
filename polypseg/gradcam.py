"""Grad-CAM for binary segmentation models.

For segmentation there is no single class score, so the target is the sum of the output
logits inside the predicted polyp region (or a user-supplied region). Gradients of that
target w.r.t. a feature layer are average-pooled into channel weights, as in
Selvaraju et al. 2017 ("Grad-CAM: Visual Explanations from Deep Networks").
"""

import cv2
import numpy as np
import torch

from .config import IMAGE_SIZE
from .inference import preprocess
from .models import GRADCAM_LAYERS


def _get_module(model, dotted: str):
    mod = model
    for part in dotted.split("."):
        mod = getattr(mod, part)
    return mod


def candidate_layers(model):
    """Top-level child names that can be used as Grad-CAM targets."""
    return [name for name, _ in model.named_children()]


def grad_cam(model, img_rgb: np.ndarray, layer: str = None, model_name: str = None,
             image_size=IMAGE_SIZE, region: np.ndarray = None, threshold: float = 0.5):
    """Compute a Grad-CAM heatmap.

    Returns (cam float32 HxW in [0,1] at original resolution, prob HxW in [0,1]).
    `region` (optional, HxW binary at original resolution) restricts the explained target.
    """
    layer = layer or GRADCAM_LAYERS.get(model_name, candidate_layers(model)[-2])
    target_module = _get_module(model, layer)
    device = next(model.parameters()).device

    store = {}

    def fwd_hook(_m, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = target_module.register_forward_hook(fwd_hook)
    was_training = model.training
    model.eval()
    try:
        x = preprocess(img_rgb, image_size).to(device).requires_grad_(True)
        logits = model(x)  # 1x1xHxW
        prob_small = torch.sigmoid(logits)[0, 0]

        if region is not None:
            sel = cv2.resize(region.astype(np.uint8), tuple(image_size), interpolation=cv2.INTER_NEAREST)
            sel = torch.from_numpy(sel).bool().to(device)
        else:
            sel = prob_small > threshold
        if sel.sum() == 0:  # nothing predicted: explain the whole output
            sel = torch.ones_like(prob_small, dtype=torch.bool)

        target = logits[0, 0][sel].sum()
        model.zero_grad(set_to_none=True)
        target.backward()

        act, grad = store["act"].detach(), store["grad"].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1))[0].float().cpu().numpy()
    finally:
        handle.remove()
        model.train(was_training)

    h, w = img_rgb.shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    prob = cv2.resize(prob_small.detach().float().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
    return cam.astype(np.float32), prob.astype(np.float32)


def heatmap_overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (img_rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)


def cam_focus_stats(cam: np.ndarray, mask: np.ndarray):
    """How much of the explanation energy falls inside the predicted mask (0-1)."""
    total = cam.sum() + 1e-8
    inside = cam[mask.astype(bool)].sum()
    return float(inside / total)


@torch.no_grad()
def occlusion_map(model, img_rgb: np.ndarray, image_size=IMAGE_SIZE, patch=16, stride=8,
                  threshold=0.5, batch=32):
    """Occlusion sensitivity (Zeiler & Fergus 2014), a model-agnostic check on Grad-CAM.

    Slides a grey patch over the input and records how much the polyp score (sum of
    probabilities inside the original predicted region) drops. Returns HxW in [0,1].
    """
    device = next(model.parameters()).device
    model.eval()
    x = preprocess(img_rgb, image_size).to(device)
    base_prob = torch.sigmoid(model(x))[0, 0]
    region = base_prob > threshold
    if region.sum() == 0:
        region = torch.ones_like(region)
    base_score = base_prob[region].sum()

    H, W = x.shape[2:]
    coords = [(y, xx) for y in range(0, H - patch + 1, stride) for xx in range(0, W - patch + 1, stride)]
    heat = torch.zeros(H, W, device=device)
    count = torch.zeros(H, W, device=device)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        xb = x.repeat(len(chunk), 1, 1, 1)
        for j, (y, xx) in enumerate(chunk):
            xb[j, :, y:y + patch, xx:xx + patch] = 0.5
        probs = torch.sigmoid(model(xb))[:, 0]
        drops = (base_score - probs[:, region].sum(dim=1)).clamp(min=0)
        for j, (y, xx) in enumerate(chunk):
            heat[y:y + patch, xx:xx + patch] += drops[j]
            count[y:y + patch, xx:xx + patch] += 1
    heat = (heat / count.clamp(min=1)).cpu().numpy()
    h, w = img_rgb.shape[:2]
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    return ((heat - heat.min()) / (heat.max() - heat.min() + 1e-8)).astype(np.float32)

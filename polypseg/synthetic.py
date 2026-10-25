"""Procedural synthetic colonoscopy-like images with polyps.

This is NOT a generative model of real anatomy. It draws a pink "tunnel" background,
blood vessels, noise and one or more shaded elliptical "polyps" with highlights.
Its uses are:
  * smoke-testing the training / inference / video pipelines without real data,
  * the "with vs without synthetic data" augmentation experiment,
  * a controllable difficulty knob for curriculum learning.
"""

from pathlib import Path

import cv2
import numpy as np


def _background(h, w, rng):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.hypot(w, h)
    base = np.array([rng.uniform(170, 220), rng.uniform(80, 120), rng.uniform(80, 115)], np.float32)
    # Dark lumen in the centre, brighter walls outward
    shade = np.clip(0.35 + 1.6 * r, 0.3, 1.15)[..., None]
    img = base[None, None, :] * shade

    # Mucosal folds
    for _ in range(rng.integers(2, 5)):
        rad = rng.uniform(0.15, 0.6) * min(h, w)
        cv2.ellipse(img, (int(cx), int(cy)), (int(rad), int(rad * rng.uniform(0.6, 1.0))),
                    rng.uniform(0, 180), 0, 360, (base * 0.75).tolist(), int(rng.integers(2, 6)))

    # Blood vessels
    for _ in range(rng.integers(4, 10)):
        pts = np.cumsum(rng.normal(0, w * 0.04, (6, 2)), axis=0) + rng.uniform(0, [w, h])
        cv2.polylines(img, [pts.astype(np.int32)], False, (150.0, 40.0, 50.0), 1, cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    return img


def generate_sample(h=256, w=256, rng=None, difficulty=None, n_polyps=None):
    """Create one synthetic image.

    difficulty in [0, 1]: 0 = big, high-contrast polyp; 1 = small, low-contrast polyp.
    Returns (rgb uint8 HxWx3, mask uint8 HxW, boxes [[x1,y1,x2,y2],...], difficulty).
    """
    rng = rng if rng is not None else np.random.default_rng()
    difficulty = float(rng.uniform(0, 1) if difficulty is None else np.clip(difficulty, 0, 1))
    n_polyps = int(n_polyps if n_polyps is not None else (1 if rng.uniform() < 0.8 else 2))

    img = _background(h, w, rng)
    mask = np.zeros((h, w), np.uint8)
    boxes = []

    for _ in range(n_polyps):
        size_frac = np.interp(difficulty, [0, 1], [0.22, 0.06]) * rng.uniform(0.8, 1.2)
        ax = max(4, int(size_frac * min(h, w)))
        ay = max(4, int(ax * rng.uniform(0.7, 1.2)))
        cx = int(rng.uniform(ax + 2, w - ax - 2))
        cy = int(rng.uniform(ay + 2, h - ay - 2))
        angle = rng.uniform(0, 180)

        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (cx, cy), (ax, ay), angle, 0, 360, 1, -1)

        # Polyp colour: shifted from local background; contrast drops with difficulty
        contrast = np.interp(difficulty, [0, 1], [1.0, 0.25])
        local = img[m.astype(bool)].mean(axis=0)
        target = np.array([235.0, 150.0, 140.0])
        colour = local + contrast * (target - local)

        # Dome shading (bright centre, darker rim) + specular highlight
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
        dome = np.clip(1.15 - 0.45 * d, 0.6, 1.15)[..., None]
        polyp = colour[None, None, :] * dome
        alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)[..., None]
        img = img * (1 - alpha) + polyp * alpha

        hx, hy = int(cx - ax * 0.3), int(cy - ay * 0.3)
        cv2.circle(img, (hx, hy), max(1, ax // 6), (255.0, 250.0, 245.0), -1, cv2.LINE_AA)

        mask = np.maximum(mask, m)
        ys, xs = np.where(m > 0)
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

    noise = rng.normal(0, np.interp(difficulty, [0, 1], [3, 9]), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, mask, boxes, difficulty


def generate_arrays(n=200, h=256, w=256, seed=42, difficulty=None):
    """Generate n samples as stacked arrays. Returns images, masks, boxes, difficulties."""
    rng = np.random.default_rng(seed)
    imgs, masks, boxes, diffs = [], [], [], []
    for _ in range(n):
        im, m, b, d = generate_sample(h, w, rng, difficulty)
        imgs.append(im)
        masks.append(m)
        boxes.append(b)
        diffs.append(d)
    return np.stack(imgs), np.stack(masks), boxes, np.array(diffs)


def export_ldpolyp_format(out_dir, n=200, sequences=4, h=256, w=256, seed=42):
    """Write synthetic data in the LDPolypVideo folder layout so the real loader can read it.

    <out_dir>/Images/<seq>/<frame>.jpg
    <out_dir>/Annotations/<seq>/<frame>.txt   (count, then "x1 y1 x2 y2" lines)
    Also writes <out_dir>/Masks/<seq>/<frame>.png with the exact elliptical masks.
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    per_seq = max(1, n // sequences)
    written = 0
    for s in range(sequences):
        seq = f"{s + 1:03d}"
        for sub in ("Images", "Annotations", "Masks"):
            (out_dir / sub / seq).mkdir(parents=True, exist_ok=True)
        for f in range(per_seq):
            img, mask, boxes, _ = generate_sample(h, w, rng)
            name = f"{f + 1:05d}"
            cv2.imwrite(str(out_dir / "Images" / seq / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "Masks" / seq / f"{name}.png"), mask * 255)
            with (out_dir / "Annotations" / seq / f"{name}.txt").open("w") as fh:
                fh.write(f"{len(boxes)}\n")
                for b in boxes:
                    fh.write(" ".join(str(v) for v in b) + "\n")
            written += 1
    return written


def write_video(path, n_frames=90, h=256, w=256, fps=15, seed=7):
    """Write a short synthetic 'colonoscopy' clip with a drifting polyp, for the video demo."""
    rng = np.random.default_rng(seed)
    base, mask, _, _ = generate_sample(h * 2, w * 2, rng, difficulty=0.2, n_polyps=1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for t in range(n_frames):
        # Pan a window across the larger image to simulate camera motion
        ox = int((w / 2) * (1 + np.sin(2 * np.pi * t / n_frames)) / 1.0)
        oy = int((h / 2) * (1 + np.cos(2 * np.pi * t / n_frames)) / 1.0)
        ox, oy = min(ox, w), min(oy, h)
        frame = base[oy:oy + h, ox:ox + w]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path

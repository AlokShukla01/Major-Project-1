"""Curriculum learning helpers.

Difficulty of a sample is estimated from its mask and image only (no extra labels):
  * small polyps are harder than large ones,
  * low colour contrast between polyp and surrounding tissue is harder.

Training then starts on the easiest fraction of the data and linearly grows to the
full set ("baby steps" pacing).
"""

import cv2
import numpy as np


def sample_difficulty(img_rgb: np.ndarray, mask: np.ndarray):
    """Return (area_fraction, contrast) for one sample. Empty masks get area 0, contrast 0."""
    m = mask.astype(bool)
    area = float(m.mean())
    if not m.any():
        return 0.0, 0.0
    kernel = np.ones((15, 15), np.uint8)
    ring = cv2.dilate(m.astype(np.uint8), kernel, iterations=2).astype(bool) & ~m
    if not ring.any():
        return area, 1.0
    inside = img_rgb[m].astype(np.float32).mean(axis=0)
    outside = img_rgb[ring].astype(np.float32).mean(axis=0)
    contrast = float(np.linalg.norm(inside - outside) / (255.0 * np.sqrt(3)))
    return area, contrast


def _rank01(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    order = values.argsort().argsort()
    return order / (len(values) - 1)


def score_dataset(dataset, indices=None, area_weight=0.6):
    """Difficulty score in [0, 1] per sample (higher = harder).

    `dataset` must expose load_raw(idx) -> (rgb uint8, mask uint8), like the loaders in
    polypseg.data. Scores are rank-normalised so the scale is dataset independent.
    """
    indices = list(range(len(dataset))) if indices is None else list(indices)
    areas, contrasts = [], []
    for i in indices:
        img, mask = dataset.load_raw(i)
        a, c = sample_difficulty(img, mask)
        areas.append(a)
        contrasts.append(c)
    areas, contrasts = np.array(areas), np.array(contrasts)
    # Small area -> hard, low contrast -> hard
    hard_area = 1.0 - _rank01(areas)
    hard_contrast = 1.0 - _rank01(contrasts)
    return area_weight * hard_area + (1 - area_weight) * hard_contrast


def pacing_fraction(epoch: int, total_epochs: int, start_fraction=0.3, full_at=0.6):
    """Fraction of the (sorted) training data to use at a given 0-based epoch.

    Grows linearly from `start_fraction` to 1.0, reaching 1.0 after `full_at` of the epochs.
    """
    ramp = max(1, int(round(total_epochs * full_at)))
    if epoch >= ramp:
        return 1.0
    return start_fraction + (1.0 - start_fraction) * epoch / ramp


def curriculum_indices(scores: np.ndarray, epoch: int, total_epochs: int, **kwargs):
    """Positions (into `scores`) of the easiest samples allowed at this epoch."""
    frac = pacing_fraction(epoch, total_epochs, **kwargs)
    n = max(1, int(round(len(scores) * frac)))
    return np.argsort(scores)[:n]

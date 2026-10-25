"""Dataset loaders.

SUNPolypDataset and LDPolypDataset are the loaders from dataset_comparison.ipynb.
Both turn bounding boxes into rectangular pseudo-masks.

ArrayDataset wraps in-memory numpy arrays (used for synthetic data).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import IMAGE_SIZE


def read_rgb_image(img_path):
    """Read an image with OpenCV and return RGB format."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_box_to_mask(mask, box):
    """Add one xmin,ymin,xmax,ymax bounding box to a binary mask."""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
    x_min, x_max = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y_min, y_max = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x_max > x_min and y_max > y_min:
        mask[y_min:y_max, x_min:x_max] = 1


def to_tensors(img_rgb: np.ndarray, mask: np.ndarray, image_size=IMAGE_SIZE):
    """Resize + normalise an RGB uint8 image and binary mask into model tensors."""
    img = cv2.resize(img_rgb, image_size).astype(np.float32) / 255.0
    mask = cv2.resize(mask.astype(np.uint8), image_size, interpolation=cv2.INTER_NEAREST)
    img = np.transpose(img, (2, 0, 1))
    return (
        torch.tensor(img, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
    )


class BoundingBoxSegmentationDataset(Dataset):
    """Common image + bounding-box pseudo-mask dataset behavior."""

    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size
        self.samples = []  # list of (img_path, [[x1, y1, x2, y2], ...])

    def __len__(self):
        return len(self.samples)

    def load_raw(self, idx):
        """Return the full-resolution RGB image and box mask (uint8)."""
        img_path, boxes = self.samples[idx]
        img = read_rgb_image(img_path)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for box in boxes:
            add_box_to_mask(mask, box)
        return img, mask

    def __getitem__(self, idx):
        img, mask = self.load_raw(idx)
        return to_tensors(img, mask, self.image_size)


class SUNPolypDataset(BoundingBoxSegmentationDataset):
    """SUN Database. Annotation format: "filename xmin,ymin,xmax,ymax,label"."""

    def __init__(self, root_dir, image_size=IMAGE_SIZE):
        super().__init__(image_size=image_size)
        root_dir = Path(root_dir)
        anno_dir = root_dir / "annotation_txt"
        if not anno_dir.exists():
            raise FileNotFoundError(f"SUN annotation folder not found: {anno_dir}")
        for txt_file in sorted(anno_dir.glob("*.txt")):
            case_dir = root_dir / txt_file.stem
            if not case_dir.exists():
                continue
            with txt_file.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    coords = parts[1].split(",")
                    if len(coords) < 4:
                        continue
                    img_path = case_dir / parts[0]
                    if img_path.exists():
                        self.samples.append((img_path, [[int(float(c)) for c in coords[:4]]]))


class LDPolypDataset(BoundingBoxSegmentationDataset):
    """LDPolypVideo.

    Layout: <root>/Images/<seq>/*.jpg and <root>/Annotations/<seq>/<frame>.txt
    Annotation file: line 0 = number of polyps, lines 1+ = "xmin ymin xmax ymax".
    """

    def __init__(self, root_dir, image_size=IMAGE_SIZE, skip_empty=True):
        super().__init__(image_size=image_size)
        root_dir = Path(root_dir)
        img_root = root_dir / "Images"
        anno_root = root_dir / "Annotations"
        if not img_root.exists():
            raise FileNotFoundError(f"LDPolyp image folder not found: {img_root}")
        if not anno_root.exists():
            raise FileNotFoundError(f"LDPolyp annotation folder not found: {anno_root}")
        for img_dir in sorted([p for p in img_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            anno_dir = anno_root / img_dir.name
            if not anno_dir.exists():
                continue
            for img_file in sorted(img_dir.glob("*.jpg")):
                anno_file = anno_dir / f"{img_file.stem}.txt"
                if not anno_file.exists():
                    continue
                with anno_file.open("r") as f:
                    lines = [line.strip().replace("\r", "") for line in f if line.strip()]
                try:
                    n_polyps = int(lines[0]) if lines else 0
                except ValueError:
                    n_polyps = 0
                if skip_empty and n_polyps == 0:
                    continue
                boxes = []
                for line in lines[1:n_polyps + 1]:
                    coords = line.replace(",", " ").split()
                    if len(coords) >= 4:
                        boxes.append([int(float(c)) for c in coords[:4]])
                if boxes or not skip_empty:
                    self.samples.append((img_file, boxes))


class ArrayDataset(Dataset):
    """In-memory dataset from uint8 RGB images (N,H,W,3) and binary masks (N,H,W)."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, image_size=IMAGE_SIZE):
        assert len(images) == len(masks), "images and masks must have the same length"
        self.images = images
        self.masks = masks
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def load_raw(self, idx):
        return self.images[idx], self.masks[idx]

    def __getitem__(self, idx):
        return to_tensors(self.images[idx], self.masks[idx], self.image_size)


DATASET_TYPES = {
    "LDPolypVideo format": LDPolypDataset,
    "SUN format": SUNPolypDataset,
}


def load_dataset(kind: str, root: str, image_size=IMAGE_SIZE):
    """Load a dataset folder by format name ("LDPolypVideo format" or "SUN format")."""
    if kind not in DATASET_TYPES:
        raise KeyError(f"Unknown dataset type '{kind}'")
    return DATASET_TYPES[kind](root, image_size=image_size)

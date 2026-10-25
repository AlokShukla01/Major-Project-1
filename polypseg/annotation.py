"""Collaborative bounding-box annotation store.

Layout of a project at data/annotations/<project>/:
    images/<name>              uploaded images
    annotations.jsonl          one line per submission: {image, annotator, boxes, time}

Several annotators can label the same image. `agreement()` reports the mean pairwise IoU
between annotators, and `export_ldpolyp()` writes a consensus set in the LDPolypVideo layout
so the existing LDPolypDataset loader can train on it directly.
"""

import datetime as dt
import json
import re
import shutil
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from . import config as C

ANNOT_ROOT = C.DATA_DIR / "annotations"
SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """Strip path components and unsafe characters (prevents path traversal)."""
    name = SAFE.sub("_", Path(name).name).strip("._")
    return name or "unnamed"


class AnnotationProject:
    def __init__(self, name: str, root=ANNOT_ROOT):
        self.name = safe_name(name)
        self.dir = Path(root) / self.name
        self.images_dir = self.dir / "images"
        self.log = self.dir / "annotations.jsonl"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ---------- images ----------
    def add_image(self, filename: str, data: bytes) -> str:
        arr = np.frombuffer(data, np.uint8)
        if cv2.imdecode(arr, cv2.IMREAD_COLOR) is None:
            raise ValueError(f"{filename} is not a readable image")
        name = safe_name(filename)
        (self.images_dir / name).write_bytes(data)
        return name

    def images(self):
        return sorted(p.name for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def image_path(self, name: str) -> Path:
        return self.images_dir / safe_name(name)

    # ---------- annotations ----------
    def submit(self, image: str, annotator: str, boxes):
        """boxes: list of [x1, y1, x2, y2] in original pixel coordinates."""
        rec = {
            "image": safe_name(image),
            "annotator": annotator.strip() or "anonymous",
            "boxes": [[int(v) for v in b] for b in boxes],
            "time": dt.datetime.now().isoformat(timespec="seconds"),
        }
        with self.log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def records(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def latest_by_annotator(self, image: str):
        """{annotator: boxes} keeping each annotator's most recent submission for an image."""
        out = {}
        for r in self.records():
            if r["image"] == safe_name(image):
                out[r["annotator"]] = r["boxes"]
        return out

    def progress(self):
        imgs = self.images()
        labelled = {r["image"] for r in self.records()}
        annotators = {r["annotator"] for r in self.records()}
        return {"images": len(imgs), "labelled": len([i for i in imgs if i in labelled]),
                "annotators": len(annotators), "submissions": len(self.records())}

    # ---------- quality ----------
    def _box_mask(self, image, boxes):
        h, w = cv2.imread(str(self.image_path(image))).shape[:2]
        m = np.zeros((h, w), np.uint8)
        for x1, y1, x2, y2 in boxes:
            m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 1
        return m

    def agreement(self, image: str):
        """Mean pairwise IoU between annotators (None if fewer than 2 annotators)."""
        by = self.latest_by_annotator(image)
        if len(by) < 2:
            return None
        masks = [self._box_mask(image, b) for b in by.values()]
        ious = []
        for a, b in combinations(masks, 2):
            union = np.logical_or(a, b).sum()
            ious.append(1.0 if union == 0 else np.logical_and(a, b).sum() / union)
        return float(np.mean(ious))

    def consensus_boxes(self, image: str, min_votes=None):
        """Pixel-vote consensus: keep pixels marked by at least half the annotators, then box them."""
        by = self.latest_by_annotator(image)
        if not by:
            return []
        if len(by) == 1:
            return next(iter(by.values()))
        votes = sum(self._box_mask(image, b).astype(np.int32) for b in by.values())
        need = min_votes or int(np.ceil(len(by) / 2))
        m = (votes >= need).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(m)
        return [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h, _ in stats[1:n]]

    # ---------- export ----------
    def export_ldpolyp(self, out_dir=None, sequence="001"):
        out_dir = Path(out_dir) if out_dir else C.DATA_DIR / "exports" / self.name
        img_dir, ann_dir = out_dir / "Images" / sequence, out_dir / "Annotations" / sequence
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in self.images():
            boxes = self.consensus_boxes(name)
            if not self.latest_by_annotator(name):
                continue
            stem = Path(name).stem
            img = cv2.imread(str(self.image_path(name)))
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
            with (ann_dir / f"{stem}.txt").open("w") as f:
                f.write(f"{len(boxes)}\n")
                for b in boxes:
                    f.write(" ".join(map(str, b)) + "\n")
            n += 1
        return out_dir, n

    def delete(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def list_projects(root=ANNOT_ROOT):
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []

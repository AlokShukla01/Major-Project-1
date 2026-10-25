"""Frame-by-frame polyp detection on colonoscopy videos.

CLI
---
    python -m polypseg.video input.mp4 --checkpoint checkpoints/deeplabv3__sun-database__baseline.pt
    python -m polypseg.video --make-demo outputs/demo_synthetic.mp4   # create a synthetic test clip

Each frame is segmented, connected components become bounding boxes, and an annotated
video plus a per-frame CSV log are written.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .inference import draw_boxes, heuristic_prob, load_checkpoint, mask_to_boxes, overlay_mask, predict_prob


def _open_writer(path, fps, size):
    """Try browser-friendly H.264 first, then fall back to mp4v."""
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, fourcc
    raise RuntimeError(f"Could not open a video writer for {path}")


def process_video(in_path, out_path, predictor, threshold=0.5, min_area=50, every_n=1,
                  show_mask=True, max_frames=None, progress_callback=None, keyframe_count=6):
    """Annotate a video.

    predictor(rgb uint8 HxWx3) -> probability map HxW in [0,1].
    Returns a dict with stats, per-frame records and a few annotated keyframes (RGB arrays).
    """
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, codec = _open_writer(out_path, fps, (w, h))

    records, keyframes = [], []
    last_boxes, last_mask = [], np.zeros((h, w), np.uint8)
    idx, t0 = 0, time.time()
    key_every = max(1, (total or 60) // max(1, keyframe_count))
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if idx % every_n == 0:
                prob = predictor(rgb)
                last_mask = (prob >= threshold).astype(np.uint8)
                last_boxes = mask_to_boxes(last_mask, prob, min_area=min_area)
            out = overlay_mask(rgb, last_mask, 0.35) if show_mask else rgb
            out = draw_boxes(out, last_boxes)
            cv2.putText(out, f"frame {idx}  polyps {len(last_boxes)}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

            records.append({
                "frame": idx,
                "time_s": round(idx / fps, 3),
                "polyps": len(last_boxes),
                "max_score": round(max((b["score"] for b in last_boxes), default=0.0), 4),
                "mask_coverage": round(float(last_mask.mean()), 5),
                "boxes": ";".join(",".join(map(str, b["box"])) for b in last_boxes),
            })
            if idx % key_every == 0 and len(keyframes) < keyframe_count:
                keyframes.append((idx, out))
            idx += 1
            if progress_callback and total:
                progress_callback(min(idx / total, 1.0))
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    with_polyp = sum(1 for r in records if r["polyps"] > 0)
    return {
        "frames": idx,
        "fps_in": fps,
        "processing_fps": idx / elapsed if elapsed else 0.0,
        "frames_with_polyp": with_polyp,
        "detection_rate": with_polyp / idx if idx else 0.0,
        "codec": codec,
        "records": records,
        "keyframes": keyframes,
        "output": str(out_path),
    }


def save_records(records, path):
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["frame"])
        w.writeheader()
        w.writerows(records)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Polyp detection on a video")
    p.add_argument("video", nargs="?", help="input video path")
    p.add_argument("--checkpoint", help="trained checkpoint (.pt). Without it a colour heuristic is used.")
    p.add_argument("--out", help="output video path (default outputs/<name>_annotated.mp4)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--every-n", type=int, default=1, help="run the model every N frames")
    p.add_argument("--make-demo", metavar="PATH", help="write a synthetic demo clip and exit")
    args = p.parse_args(argv)

    if args.make_demo:
        from .synthetic import write_video
        Path(args.make_demo).parent.mkdir(parents=True, exist_ok=True)
        print(f"Wrote {write_video(args.make_demo)}")
        return
    if not args.video:
        p.error("video path is required (or use --make-demo)")

    if args.checkpoint:
        model, meta = load_checkpoint(args.checkpoint)
        predictor = lambda rgb: predict_prob(model, rgb, meta["image_size"])  # noqa: E731
    else:
        print("WARNING: no checkpoint given, using untrained colour heuristic.")
        predictor = heuristic_prob

    out = Path(args.out) if args.out else C.OUTPUT_DIR / f"{Path(args.video).stem}_annotated.mp4"
    stats = process_video(args.video, out, predictor, args.threshold, args.min_area, args.every_n)
    log = save_records(stats["records"], out.with_suffix(".csv"))
    print(f"{stats['frames']} frames | polyp in {stats['frames_with_polyp']} | "
          f"{stats['processing_fps']:.1f} fps | codec {stats['codec']}")
    print(f"Video: {out}\nLog:   {log}")


if __name__ == "__main__":
    main()

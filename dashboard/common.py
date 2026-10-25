"""Shared Streamlit helpers for the extended dashboard pages."""

import io
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from polypseg import config as C
from polypseg.inference import list_checkpoints, load_checkpoint
from polypseg.synthetic import generate_sample

HEURISTIC = "Colour heuristic (no trained model)"


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "upload"


@st.cache_resource(show_spinner="Loading model...")
def _load(path_str: str, mtime: float):
    return load_checkpoint(path_str)


def get_checkpoints():
    return list_checkpoints()


def checkpoint_picker(key: str, allow_heuristic=True, label="Model checkpoint"):
    """Select box over saved checkpoints. Returns (model, meta) or (None, None) for the heuristic."""
    cks = get_checkpoints()
    options = [c["label"] for c in cks] + ([HEURISTIC] if allow_heuristic else [])
    if not options:
        st.info("No trained checkpoints yet. Train one on the **Train Model** page "
                "(synthetic data works without any dataset).")
        return None, None
    choice = st.selectbox(label, options, key=key)
    if choice == HEURISTIC:
        return None, None
    ck = cks[options.index(choice)]
    model, meta = _load(str(ck["path"]), ck["path"].stat().st_mtime)
    m = ck["metrics"]
    st.caption(f"Trained on **{ck['dataset']}** ({ck['variant']}) | val Dice {m.get('final_dice', 0):.4f} | "
               f"{(ck.get('parameters') or 0):,} params")
    return model, meta


def image_source(key: str, allow_synthetic=True):
    """Upload widget + 'synthetic sample' option. Returns (rgb uint8, ground-truth mask or None, name)."""
    choices = ["Upload image"] + (["Synthetic sample"] if allow_synthetic else [])
    src = st.radio("Image source", choices, horizontal=True, key=f"{key}_src")
    if src == "Upload image":
        up = st.file_uploader("Colonoscopy image", type=["png", "jpg", "jpeg"], key=f"{key}_up")
        if up is None:
            return None, None, None
        return np.asarray(Image.open(up).convert("RGB")), None, up.name
    c1, c2 = st.columns(2)
    seed = c1.number_input("Seed", 0, 10_000, 3, key=f"{key}_seed")
    diff = c2.slider("Difficulty", 0.0, 1.0, 0.4, 0.05, key=f"{key}_diff",
                     help="0 = large, high-contrast polyp. 1 = small, low-contrast polyp.")
    img, mask, _, _ = generate_sample(256, 256, np.random.default_rng(int(seed)), difficulty=diff)
    return img, mask, f"synthetic_{seed}.png"


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def save_upload(uploaded, subdir="uploads") -> Path:
    out = C.OUTPUT_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / safe_filename(uploaded.name)
    path.write_bytes(uploaded.getbuffer())
    return path


def temp_path(suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=C.OUTPUT_DIR if C.OUTPUT_DIR.exists() else None)
    f.close()
    return Path(f.name)


def colourise(prob: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

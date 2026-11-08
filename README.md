# Major-Project-1: PolypSeg AI

Deep learning polyp segmentation in colonoscopy images (SUN Database, LDPolypVideo), with an
analytics and research platform around it. See `CompleteIdea` for the full feature list.

> Research prototype. Not a medical device and not for clinical use.

## Project status

| Part | Status |
|---|---|
| Active dataset | LDPolypVideo only (`TrainValid` for training, `Test` for final evaluation) |
| Original notebook comparison | Historical SUN and LDPolypVideo results remain in the notebook/CSV |
| Dashboard, training pipeline, curriculum learning, Grad-CAM, video detection, synthetic data + DCGAN, annotation tool, leaderboard, knowledge graph, RAG assistant | Implemented; active real-data workflow is LDPolypVideo |
| Dataset folders | Local-only files in `LD Polyp/`; ignored by Git |
| Selected video model | Fine-tuned Polyp-PVT, default in Video Polyp Detection |
| Full held-out Test | 15,397 frames: Dice 0.5001, IoU 0.4110 at threshold 0.90 |
| Image labels | LDPolypVideo bounding boxes become rectangular pseudo-masks; not pixel-accurate contours |
| Idea 6, severity scoring | Not built: available datasets have no severity labels |

Not tracked by Git: `checkpoints/` (except the selected Polyp-PVT checkpoint via Git LFS), `runs/`,
`outputs/`, `data/`, `leaderboard.csv`, `.env`, the dataset (`LD Polyp/`), and videos. The old SUN
notebook results remain as historical reference.

## Setup

```bash
git clone https://github.com/AlokShukla01/Major-Project-1.git
cd Major-Project-1
# Install Git LFS for your operating system before running these commands
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Connect the datasets

### 1. Place the datasets

The active project workflow uses LDPolypVideo. Train on TrainValid and reserve Test for final evaluation:

```
Major-Project-1/
└── LD Polyp/
    ├── TrainValid/                      # Images/<seq>/*.jpg and Annotations/<seq>/*.txt
    └── Test/                            # same layout
```

The current checkout's loader finds **24,789** TrainValid frames (20,942 polyp frames and 3,847
background frames) and **15,397** Test frames (12,933 polyp frames and 2,464 background frames).
These local files are not included in Git.

If the folders are elsewhere, pass their full paths. If you only have the LDPolyp `TrainValid` archive,
extract it so `TrainValid/Images/` and `TrainValid/Annotations/` exist. The `Test` split is separate and should
be kept for final evaluation, not used as the training folder. The loader turns each annotated bounding
box into a rectangular pseudo-mask, not a pixel-accurate mask.

### 2. Check the loaders find the data

```bash
python -m polypseg.data --ldpolyp "LD Polyp/TrainValid"
python -m polypseg.data --ldpolyp "LD Polyp/Test"
```

The command prints the number of images and boxes the loader actually found. The dashboard's
**Train Model → Dataset folder** page defaults to the local TrainValid path.

### 3. Train real checkpoints

The selected video model is Polyp-PVT, initialized from its official pretrained weights and fine-tuned on LDPolypVideo. The run used 384x384 inputs, batch size 4, learning rate 1e-5, one epoch, and a sequence-grouped 80/20 split within TrainValid. The model's probability threshold was calibrated on the held-out TrainValid sequences; the Test split remained untouched until final evaluation.

```bash
# Fine-tune the official Polyp-PVT checkpoint (place/import official weights first)
python -m polypseg.import_polyp_pvt --weights checkpoints/PolypPVT_official.pth
python -m polypseg.train --data "ldpolyp:LD Polyp/TrainValid" --models "Polyp-PVT" --epochs 1 --subset 1000000 --batch-size 4 --lr 1e-5 --image-size 384 --augment --init-checkpoint checkpoints/polyp-pvt__official-zero-shot.pt --run-tag polyp-pvt-video-epoch1

# Calibrate on held-out TrainValid sequences only
python -m polypseg.calibrate checkpoints/polyp-pvt__ldpolypvideo__polyp-pvt-video-epoch1.pt --data "LD Polyp/TrainValid"

# Optional: other architectures and project experiments remain available
python -m polypseg.train --data "ldpolyp:LD Polyp/TrainValid" --models all --epochs 5 --subset 1000 --base 16
python -m polypseg.train --data "ldpolyp:LD Polyp/TrainValid" --models "Hybrid Polyp U-Net" --curriculum --epochs 5 --subset 1000 --base 16 --image-size 256 --augment
tensorboard --logdir runs
```

The selected checkpoint, `checkpoints/polyp-pvt__ldpolypvideo__polyp-pvt-video-epoch1.pt`, is tracked with Git LFS and downloaded by `git lfs pull`. Other checkpoints and the original official weights are excluded from Git. The training dataset is not needed for inference; it is required only for training and evaluation.

Every run writes:
- a checkpoint, `checkpoints/<model>__<dataset>__<variant>.pt`
- a row in `outputs/pipeline_results.csv`, with the same columns as `model_comparison_results.csv`
- TensorBoard logs in `runs/`

Re-running the same model, dataset and variant overwrites that checkpoint.

You can also do all of this from the dashboard: **Train Model → Dataset folder**.

### 4. Evaluate on the held-out LDPolypVideo Test split

```bash
# Full Test evaluation for the selected checkpoint
python -m polypseg.evaluate --data "LD Polyp/Test" --checkpoint checkpoints/polyp-pvt__ldpolypvideo__polyp-pvt-video-epoch1.pt --max-samples 0 --batch-size 4 --output outputs/polyp_pvt_finetuned_full_test.csv
```

The selected Polyp-PVT checkpoint scored **Dice 0.5001 / IoU 0.4110** on all 15,397 Test frames at threshold 0.90. The official pretrained checkpoint without LDPolypVideo fine-tuning scored Dice 0.3701 / IoU 0.2862 on the same frames. Calibrated validation Dice was 0.7865 on 5,234 held-out TrainValid frames; that validation score is not the Test score. These metrics compare predictions with rectangular pseudo-masks rasterized from dataset bounding boxes, not pixel-accurate outlines. The requested Dice 0.85 target was not reached.

### 5. Play an annotated video

The video page processes an upload, then plays the generated video with a box and confidence label
for each predicted connected polyp region. Boxes update frame by frame in playback; processing
finishes before playback starts. The CLI writes the same annotated video and a per-frame CSV:

```bash
python -m polypseg.video "path/to/colonoscopy.mp4" \
  --checkpoint checkpoints/polyp-pvt__ldpolypvideo__polyp-pvt-video-epoch1.pt
```

In Streamlit, open **Video Analysis**; the fine-tuned Polyp-PVT checkpoint is selected by default. Upload an MP4/MOV/AVI/MKV and press **Process video**. You can select another checkpoint if needed. The calibrated threshold is loaded from the checkpoint and can be adjusted.

### 6. GAN

```bash
python -m polypseg.video "path/to/colonoscopy.mp4" --checkpoint checkpoints/deeplabv3plus__ldpolypvideo__baseline.pt
python -m polypseg.gan --images "LD Polyp/TrainValid/Images" --epochs 50
```

### 6. Open the dashboard

```bash
streamlit run app.py
```

- Switch on **Include pipeline runs** in the sidebar to show the new results next to the notebook results.
- Live Demo, Explainable AI, Video Analysis and Leaderboard automatically list every checkpoint in `checkpoints/`.
- The AI Assistant re-indexes results and checkpoints on its own.

## Quick start without a dataset

```bash
python -m polypseg.train --data synthetic --synthetic-n 600 --models all --epochs 8 --base 32
python -m polypseg.train --data synthetic --synthetic-n 600 --models "Basic U-Net,DeepLabV3+" --curriculum --epochs 8 --base 32
python -m polypseg.gan --images synthetic --epochs 15
streamlit run app.py
```

This takes about 2 minutes on an M-series Mac. Synthetic checkpoints are only for demonstrating the
tools. You can delete them (`rm checkpoints/*synthetic*`) once real ones exist.

## AI Assistant (RAG) LLM providers

Copy `.env.example` to `.env` and set one key, or paste a key into the assistant page.
`.env` is git-ignored, so never commit keys.

| Provider | Default model | Free key |
|---|---|---|
| Groq (recommended) | `openai/gpt-oss-20b` | https://console.groq.com/keys |
| Google Gemini | `gemini-flash-latest` | https://aistudio.google.com/apikey |
| OpenRouter | `openrouter/free` | https://openrouter.ai/keys |
| Ollama (local) | `llama3.2` | no key, https://ollama.com |

Without a key the assistant works in retrieval-only mode. With a provider selected, your question and
the retrieved project passages are sent to that provider.

## Layout

| Path | Contents |
|---|---|
| `dataset_comparison.ipynb` | Original experiments; produces `model_comparison_results.csv` |
| `CompleteIdea` | Project ideas and implemented-feature list |
| `app.py` | Streamlit dashboard (analytics pages + routing) |
| `dashboard/` | One module per extended page |
| `polypseg/models.py` | Basic U-Net, Attention U-Net, U-Net++, DeepLabV3+, Hybrid Polyp U-Net, Pretrained ResNet18 U-Net, Polyp-PVT |
| `polypseg/data.py` | LDPolypVideo active loader + SUN compatibility loader |
| `polypseg/train.py` | Training pipeline (CLI + API, TensorBoard, checkpoints) |
| `polypseg/evaluate.py` | Shared-sample held-out Test evaluation for checkpoints |
| `polypseg/curriculum.py` | Difficulty scoring + pacing |
| `polypseg/synthetic.py`, `polypseg/gan.py` | Procedural generator, DCGAN |
| `polypseg/inference.py`, `polypseg/gradcam.py`, `polypseg/video.py` | Prediction, Grad-CAM + occlusion, video detection |
| `polypseg/annotation.py`, `polypseg/leaderboard.py` | Annotation store, benchmark leaderboard |
| `polypseg/knowledge_graph.py`, `polypseg/rag.py` | Knowledge graph (networkx, Cypher export), RAG chatbot |

# Major-Project-1: PolypSeg AI

Deep learning polyp segmentation in colonoscopy images (SUN Database, LDPolypVideo), with an
analytics and research platform around it. See `CompleteIdea` for the full feature list.

> Research prototype. Not a medical device and not for clinical use.

## Project status

| Part | Status |
|---|---|
| Notebook experiments (`dataset_comparison.ipynb`, `model_comparison_results.csv`) | Done on the real datasets |
| Dashboard, training pipeline, curriculum learning, Grad-CAM, video detection, synthetic data + DCGAN, annotation tool, leaderboard, knowledge graph, RAG assistant | Built and tested on **synthetic data only** (the datasets were not on the development machine) |
| Trained checkpoints on SUN / LDPolypVideo | **To do** on the dataset machine (see below) |
| Idea 6, severity scoring | Not built: the datasets have no severity labels |

Not in the repo (git-ignored, recreate locally): `checkpoints/`, `runs/`, `outputs/`, `data/`,
`leaderboard.csv`, `.env`, and the datasets (`Sun/`, `LD Polyp/`, videos).

## Setup

```bash
git clone https://github.com/AlokShukla01/Major-Project-1.git
cd Major-Project-1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running on the dataset machine

### 1. Place the datasets

Put the datasets in the repo root using the same folder names as the notebook. Both are git-ignored.

```
Major-Project-1/
├── Sun/sundatabase_positive_part1/      # contains annotation_txt/ and one folder per case
└── LD Polyp/
    ├── TrainValid/                      # Images/<seq>/*.jpg and Annotations/<seq>/*.txt
    └── Test/                            # same layout
```

If they live somewhere else, pass the full path in the commands below. The notebook keeps its own
hard-coded `PROJECT_ROOT` in the first code cell, so update that if your folder differs.

### 2. Check the loaders find the data

```bash
python -c "
from polypseg.data import SUNPolypDataset, LDPolypDataset
print('SUN        ', len(SUNPolypDataset('Sun/sundatabase_positive_part1')))
print('LD train   ', len(LDPolypDataset('LD Polyp/TrainValid')))
print('LD test    ', len(LDPolypDataset('LD Polyp/Test')))
"
```

### 3. Train real checkpoints

The defaults match the notebook: 128x128 images, 5 epochs, batch 8, lr 1e-3, 1000 samples, base width 64, seed 42.

```bash
# All 4 models on both datasets (reproduces the notebook, now with saved checkpoints)
python -m polypseg.train --data "sun:Sun/sundatabase_positive_part1" --models all
python -m polypseg.train --data "ldpolyp:LD Polyp/TrainValid" --models all

# Idea 10: baseline vs curriculum learning
python -m polypseg.train --data "sun:Sun/sundatabase_positive_part1" --models all --compare-curriculum
python -m polypseg.train --data "ldpolyp:LD Polyp/TrainValid" --models all --compare-curriculum

# Idea 4: baseline vs + synthetic training data
python -m polypseg.train --data "ldpolyp:LD Polyp/TrainValid" --models "U-Net++,DeepLabV3+" --mix-synthetic 500

# Watch training curves
tensorboard --logdir runs
```

Every run writes:
- a checkpoint, `checkpoints/<model>__<dataset>__<variant>.pt`
- a row in `outputs/pipeline_results.csv`, with the same columns as `model_comparison_results.csv`
- TensorBoard logs in `runs/`

Re-running the same model, dataset and variant overwrites that checkpoint.

You can also do all of this from the dashboard: **Train Model → Dataset folder**.

### 4. Evaluate on the LDPolypVideo test split (optional, for the leaderboard)

```bash
python -c "
import pandas as pd
from polypseg.data import LDPolypDataset
from polypseg.inference import list_checkpoints
from polypseg import leaderboard as LB
test = LDPolypDataset('LD Polyp/Test')
rows = [LB.evaluate_checkpoint(c['path'], dataset=test, dataset_name='LDPolypVideo Test')
        for c in list_checkpoints() if c['dataset'] == 'LDPolypVideo']
LB.add_submission(pd.DataFrame(rows), 'Project baseline', 'checkpoint', 'LDPolypVideo Test split')
print(pd.DataFrame(rows)[['model', 'final_dice', 'final_iou', 'final_accuracy']])
"
```

### 5. Real video and GAN

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
| `polypseg/models.py` | Basic U-Net, Attention U-Net, U-Net++, DeepLabV3+ |
| `polypseg/data.py` | SUN / LDPolypVideo loaders |
| `polypseg/train.py` | Training pipeline (CLI + API, TensorBoard, checkpoints) |
| `polypseg/curriculum.py` | Difficulty scoring + pacing |
| `polypseg/synthetic.py`, `polypseg/gan.py` | Procedural generator, DCGAN |
| `polypseg/inference.py`, `polypseg/gradcam.py`, `polypseg/video.py` | Prediction, Grad-CAM + occlusion, video detection |
| `polypseg/annotation.py`, `polypseg/leaderboard.py` | Annotation store, benchmark leaderboard |
| `polypseg/knowledge_graph.py`, `polypseg/rag.py` | Knowledge graph (networkx, Cypher export), RAG chatbot |

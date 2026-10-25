"""Retrieval-Augmented Generation chatbot over project knowledge.

Knowledge base (built automatically, no dataset needed):
  * every row of model_comparison_results.csv and the pipeline/leaderboard CSVs,
  * computed aggregates (best model per dataset, averages, rankings),
  * model architecture and parameter counts, dataset and metric descriptions,
  * README.md and CompleteIdea,
  * metadata of trained checkpoints.

Retrieval: TF-IDF (word 1-2 grams) + cosine similarity via scikit-learn.
Generation: any OpenAI-compatible chat endpoint. Free options are listed in PROVIDERS.
Without an API key the bot falls back to an extractive answer (the top passages).

API keys are read from environment variables or a local `.env` file (never committed).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config as C

PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-20b",
        "env": "GROQ_API_KEY",
        "signup": "https://console.groq.com/keys",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-flash-latest",
        "env": "GEMINI_API_KEY",
        "signup": "https://aistudio.google.com/apikey",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openrouter/free",
        "env": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/keys",
    },
    "Ollama (local, no key)": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "env": None,
        "signup": "https://ollama.com/download",
    },
}

SYSTEM_PROMPT = (
    "You are PolypSeg Assistant, a helpful research assistant for a deep-learning project on "
    "polyp segmentation in colonoscopy images. Answer ONLY from the provided context. "
    "Quote exact metric values with 4 decimals when relevant. If the context does not contain "
    "the answer, say so plainly. Cite sources inline like [1], [2]. Keep answers concise. "
    "You do not give medical advice or diagnoses."
)


def load_dotenv(path=C.PROJECT_ROOT / ".env"):
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Doc:
    text: str
    source: str


# ------------------------------------------------------------------ corpus

MODEL_DOCS = {
    "Basic U-Net": "Basic U-Net is a symmetric encoder-decoder with two down-sampling stages, "
                   "transposed-convolution up-sampling and skip connections that concatenate encoder "
                   "features into the decoder. It is the baseline architecture in this project.",
    "Attention U-Net": "Attention U-Net adds attention gates on the skip connections. Each gate uses the "
                       "decoder signal to weight encoder features so irrelevant background is suppressed "
                       "before concatenation.",
    "U-Net++": "U-Net++ replaces plain skip connections with nested, dense skip pathways (nodes x01, x11, "
               "x02) that reduce the semantic gap between encoder and decoder features.",
    "DeepLabV3+": "DeepLabV3+ uses an encoder, an Atrous Spatial Pyramid Pooling (ASPP) module with "
                  "dilation rates 2, 4 and 6 to capture multi-scale context, and a decoder that fuses "
                  "low-level features for sharper boundaries.",
}
DATASET_DOCS = {
    "SUN Database": "SUN Database (SUN Colonoscopy Video Database) provides colonoscopy video frames with "
                    "bounding-box annotations in annotation_txt files ('filename xmin,ymin,xmax,ymax,label'). "
                    "This project converts boxes to rectangular pseudo-masks.",
    "LDPolypVideo": "LDPolypVideo provides colonoscopy video frames with per-frame annotation files: the "
                    "first line is the number of polyps followed by 'xmin ymin xmax ymax' lines. Frames "
                    "without polyps are skipped. Boxes are converted to pseudo-masks.",
}
METRIC_DOCS = {
    "Dice": "What is the Dice score? The Dice coefficient (F1 score for masks) is defined as "
            "2|P∩T| / (|P| + |T|) where P is the predicted mask and T the ground-truth mask. It measures "
            "overlap from 0 (none) to 1 (perfect) and is the primary metric of this project.",
    "IoU": "What is IoU? IoU (Intersection over Union, also called the Jaccard index) is defined as "
           "|P∩T| / |P∪T|. It is always lower than or equal to Dice and penalises boundary errors more strongly.",
    "Accuracy": "What is pixel accuracy? Pixel accuracy is the fraction of correctly classified pixels. "
                "Because most pixels are background, accuracy is inflated and stays high (around 0.95) even "
                "for weak models, so Dice and IoU are preferred.",
    "Loss": "What is the training loss? Training uses binary cross-entropy with logits (BCEWithLogitsLoss). "
            "Predictions are passed through a sigmoid and thresholded at 0.5 to produce masks.",
}
SETUP_DOC = (
    "Experimental setup (dataset_comparison.ipynb): images resized to 128x128, batch size 8, 5 epochs, "
    "Adam optimiser with learning rate 1e-3, BCEWithLogitsLoss, 1000 samples per dataset with an 80/20 "
    "train/validation split, seed 42, base channel width 64, Apple MPS / CUDA / CPU device."
)


def _chunk(text, source, size=700):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    docs, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > size and buf:
            docs.append(Doc(buf.strip(), source))
            buf = ""
        buf += p + "\n\n"
    if buf.strip():
        docs.append(Doc(buf.strip(), source))
    return docs


def _result_docs(df: pd.DataFrame, source: str):
    docs = []
    for _, r in df.iterrows():
        docs.append(Doc(
            f"Experiment result: {r['model']} trained on {r['dataset']} achieved final Dice "
            f"{r['final_dice']:.4f}, final IoU {r['final_iou']:.4f}, final validation accuracy "
            f"{r['final_accuracy']:.4f}, best Dice {r.get('best_dice', float('nan')):.4f}, best IoU "
            f"{r.get('best_iou', float('nan')):.4f}, best training loss {r.get('best_train_loss', float('nan')):.4f}, "
            f"using {int(r.get('samples_used', 0) or 0)} samples.", source))
    if df.empty:
        return docs

    for ds, g in df.groupby("dataset"):
        g = g.sort_values("final_dice", ascending=False)
        order = ", ".join(f"{i + 1}. {m} (Dice {d:.4f})" for i, (m, d) in enumerate(zip(g["model"], g["final_dice"])))
        docs.append(Doc(f"Ranking on {ds} by final Dice: {order}. The best model on {ds} is "
                        f"{g.iloc[0]['model']} and the weakest is {g.iloc[-1]['model']}.", source))
    avg = df.groupby("model")[["final_dice", "final_iou", "final_accuracy"]].mean().sort_values("final_dice", ascending=False)
    lines = "; ".join(f"{m}: mean Dice {r.final_dice:.4f}, mean IoU {r.final_iou:.4f}, mean accuracy {r.final_accuracy:.4f}"
                      for m, r in avg.iterrows())
    docs.append(Doc(f"Average performance per model across datasets: {lines}. Overall best model on average "
                    f"Dice is {avg.index[0]}.", source))
    best = df.loc[df["final_dice"].idxmax()]
    worst = df.loc[df["final_dice"].idxmin()]
    docs.append(Doc(f"Overall best single experiment: {best['model']} on {best['dataset']} with Dice "
                    f"{best['final_dice']:.4f} and IoU {best['final_iou']:.4f}. Overall worst: {worst['model']} on "
                    f"{worst['dataset']} with Dice {worst['final_dice']:.4f}.", source))
    if {"best_dice", "final_dice"} <= set(df.columns):
        gap = (df["best_dice"] - df["final_dice"])
        unstable = df.loc[gap.idxmax()]
        docs.append(Doc(f"Training stability: the largest drop from best to final Dice is {gap.max():.4f} for "
                        f"{unstable['model']} on {unstable['dataset']} (best {unstable['best_dice']:.4f}, final "
                        f"{unstable['final_dice']:.4f}), suggesting unstable training or overfitting.", source))
    return docs


def build_corpus(extra_csvs=None):
    docs = []
    if C.RESULTS_CSV.exists():
        docs += _result_docs(pd.read_csv(C.RESULTS_CSV), "model_comparison_results.csv")
    for path in extra_csvs or []:
        p = Path(path)
        if p.exists():
            try:
                docs += _result_docs(pd.read_csv(p), p.name)
            except Exception:
                pass
    for name, text in MODEL_DOCS.items():
        docs.append(Doc(text, f"model:{name}"))
    try:
        from .models import build_model, count_parameters
        counts = ", ".join(f"{n}: {count_parameters(build_model(n)):,}" for n in MODEL_DOCS)
        docs.append(Doc(f"Trainable parameter counts at base width 64: {counts}.", "models.py"))
    except Exception:
        pass
    for name, text in DATASET_DOCS.items():
        docs.append(Doc(text, f"dataset:{name}"))
    for name, text in METRIC_DOCS.items():
        docs.append(Doc(text, f"metric:{name}"))
    docs.append(Doc(SETUP_DOC, "dataset_comparison.ipynb"))
    for fname in ["README.md", "CompleteIdea"]:
        p = C.PROJECT_ROOT / fname
        if p.exists():
            docs += _chunk(p.read_text(), fname)
    try:
        from .inference import list_checkpoints
        for ck in list_checkpoints():
            m = ck["metrics"]
            docs.append(Doc(f"Saved checkpoint {ck['path'].name}: {ck['model_name']} trained on {ck['dataset']} "
                            f"({ck['variant']}), final Dice {m.get('final_dice', 0):.4f}, IoU {m.get('final_iou', 0):.4f}, "
                            f"{ck.get('parameters') or 0:,} parameters, created {ck['created']}.", "checkpoints/"))
    except Exception:
        pass
    return docs


# ------------------------------------------------------------------ retrieval

class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.matrix = self.vec.fit_transform([d.text for d in docs])

    def search(self, query, k=5):
        q = self.vec.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in top if sims[i] > 0]


# ------------------------------------------------------------------ generation

def resolve_key(provider: str, override: str = None):
    if override:
        return override
    env = PROVIDERS[provider]["env"]
    load_dotenv()
    return os.environ.get(env) if env else None


def chat_completion(provider, messages, model=None, api_key=None, timeout=60, temperature=0.2):
    cfg = PROVIDERS[provider]
    headers = {"Content-Type": "application/json"}
    if cfg["env"]:
        if not api_key:
            raise RuntimeError(f"No API key for {provider}. Set {cfg['env']} or paste a key.")
        headers["Authorization"] = f"Bearer {api_key}"
    if provider == "OpenRouter":
        headers["X-Title"] = "PolypSeg AI"
    resp = requests.post(f"{cfg['base_url']}/chat/completions", headers=headers, timeout=timeout,
                         json={"model": model or cfg["model"], "messages": messages, "temperature": temperature})
    if resp.status_code != 200:
        raise RuntimeError(f"{provider} error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def build_messages(question, hits, history=None):
    context = "\n\n".join(f"[{i + 1}] (source: {d.source})\n{d.text}" for i, (d, _) in enumerate(hits))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return msgs


def extractive_answer(hits):
    if not hits:
        return "I could not find anything relevant in the project knowledge base."
    return "No LLM connected, so here are the most relevant passages:\n\n" + "\n\n".join(
        f"**[{i + 1}]** {d.text}" for i, (d, _) in enumerate(hits[:3]))


def answer(question, retriever, provider=None, model=None, api_key=None, history=None, k=5):
    """Return {'answer', 'sources', 'mode'} where mode is 'llm' or 'extractive'."""
    hits = retriever.search(question, k=k)
    sources = [{"n": i + 1, "source": d.source, "score": s, "text": d.text} for i, (d, s) in enumerate(hits)]
    if provider:
        key = resolve_key(provider, api_key)
        if key or PROVIDERS[provider]["env"] is None:
            text = chat_completion(provider, build_messages(question, hits, history), model, key)
            return {"answer": text, "sources": sources, "mode": "llm"}
    return {"answer": extractive_answer(hits), "sources": sources, "mode": "extractive"}

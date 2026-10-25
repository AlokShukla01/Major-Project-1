"""PolypSeg AI: reusable code for polyp segmentation experiments.

Modules
-------
config         Shared defaults and project paths.
models         Basic U-Net, Attention U-Net, U-Net++, DeepLabV3+.
metrics        Accuracy / Dice / IoU.
data           SUN and LDPolypVideo loaders plus a generic image-folder loader.
synthetic      Procedural synthetic polyp generator (no real data needed).
curriculum     Difficulty scoring and curriculum sampling.
train          Automated training pipeline (CLI + Python API) with TensorBoard logging.
inference      Checkpoint discovery, loading and mask prediction.
gradcam        Grad-CAM explanations for segmentation models.
video          Frame-by-frame video polyp detection with box overlays.
gan            Small DCGAN for synthetic image generation.
annotation     Save / load bounding-box annotations (LDPolypVideo format).
leaderboard    Benchmark submissions and ranking.
knowledge_graph  Graph linking datasets, models and metrics.
rag            Retrieval-augmented chatbot over project knowledge.
"""

__version__ = "0.2.0"

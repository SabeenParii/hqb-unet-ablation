"""Single source of truth for every setting used in the experiments.

All three models (and the ablation variants) are trained with exactly these
values, so differences between models cannot come from different settings.
"""
from dataclasses import dataclass, asdict, field
import json


@dataclass
class Config:
    # ---------------- paths (set these to your own folders) ----------------
    raw_image_dir: str = "data/raw_images"  # original mammograms (not distributed)
    work_dir: str = "results"

    # ---------------- data preparation ----------------
    image_size: int = 256          # final square side length (pixels)
    pad_to_square: bool = True     # zero-pad before resizing to keep aspect ratio
    otsu_kernel: int = 20          # elliptical kernel for opening/closing (full-resolution pixels)
    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ---------------- training (identical for every model) ----------------
    in_channels: int = 1           # grayscale input for ALL models
    batch_size: int = 16
    max_epochs: int = 50
    learning_rate: float = 1e-4
    early_stop_patience: int = 10  # on val_loss, best weights restored
    lr_patience: int = 4           # ReduceLROnPlateau on val_loss
    lr_factor: float = 0.5
    min_lr: float = 1e-7
    mixed_precision: bool = False  # fp16 caused NaN validation loss in the original Classical U-Net run
    seeds: tuple = (0, 1, 2)       # independent training runs per model

    # ---------------- online augmentation (training set only) ----------------
    aug_flip_prob: float = 0.5
    aug_rotation_deg: float = 10.0
    aug_brightness: float = 0.2    # multiplicative factor in [1-b, 1+b]
    aug_contrast: float = 0.2      # contrast factor in [1-c, 1+c]

    # ---------------- quantum bottleneck ----------------
    n_qubits: int = 4
    n_q_layers: int = 2
    quantum_backend: str = "tf"  # "tf": exact TF state-vector simulation (used for all reported runs); "pennylane": default.qubit (optional)

    # ---------------- evaluation ----------------
    threshold: float = 0.5
    n_bootstrap: int = 2000        # patient-level bootstrap resamples for 95% CIs
    auc_max_pixels: int = 2_000_000  # random pixel subsample for ROC-AUC / PR-AUC
    ece_bins: int = 15

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        if "seeds" in d:
            d["seeds"] = tuple(d["seeds"])
        return cls(**d)


# Model names used throughout the code, results files and the manuscript.
MODELS = {
    "unet": "Classical U-Net",
    "attention_unet": "Attention U-Net",
    "hybrid_quantum": "Hybrid Quantum U-Net",
    # Ablations: identical to hybrid_quantum except for the 4-dimensional bottleneck block
    "hybrid_classical": "Hybrid U-Net, classical bottleneck (ablation)",
    "hybrid_none": "Hybrid U-Net, no bottleneck block (ablation)",
}

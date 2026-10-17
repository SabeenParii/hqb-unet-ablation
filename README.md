# Does the Quantum Circuit Help? Parameter-matched ablation of a quantum-bottleneck U-Net

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23390486.svg)](https://doi.org/10.5281/zenodo.23390486)

Code for the manuscript *"Parameter-Matched Ablation of a Quantum-Bottleneck U-Net on Proxy-Labeled
Mammograms"* (S. Parii, S. Tummala, A. Iqbal, R. Ashraf, F. Aadil).

The package `qseg` trains and evaluates five models on the same patient-level split and with
identical training settings:

| Name in code       | Name in paper | Description |
|--------------------|---------------|-------------|
| `unet`             | C-UNet        | Classical U-Net (64–1024 filters) |
| `attention_unet`   | Att-UNet      | Attention U-Net (16–256 filters) |
| `hybrid_quantum`   | HQB-UNet      | U-Net with a simulated 4-qubit circuit at the globally pooled bottleneck |
| `hybrid_classical` | HQB-cls       | Ablation: circuit replaced by Dense(4, tanh) (parameter-matched) |
| `hybrid_none`      | HQB-free      | Ablation: circuit removed |

The circuit is simulated exactly (noise-free state vector) in TensorFlow (`qseg/quantum_sim.py`)
and checked against an independent NumPy reference (`qseg/quantum_reference.py`).

## Data

**The mammograms are not included.** They are private, de-identified images provided by a hospital
and cannot be redistributed; requests go to the corresponding author and need permission from the
data provider. No images, masks, model weights or patient identifiers are in this repository.
The proxy masks are generated from the images by `qseg/data_prep.py` (Otsu threshold, opening and
closing, largest contour).

## Structure

```
qseg/
  config.py             all settings (single source of truth)
  data_prep.py          de-duplication, proxy masks, patient-level split (manifest.csv)
  datasets.py           tf.data pipeline and online augmentation
  models.py             C-UNet, Att-UNet, HQB-UNet and the two ablations
  quantum_sim.py        exact state-vector simulation of the 4-qubit circuit (TF and NumPy)
  quantum_reference.py  independent NumPy reference of the circuit (and of the earlier, inert circuit)
  base_ansatz_check.py  Jacobian-rank check of our circuit and of the base-design ansatz (Lim & Kim 2026)
  train.py              training and test-set evaluation of one model and seed
  evaluation.py         pixel-level metrics, per-image Dice/IoU, ROC-AUC, PR-AUC, ECE
  final_stats.py        hierarchical and patient-only bootstrap, equivalence bounds, Wilcoxon + Holm
  stats.py, figures.py  aggregation helpers and architecture diagrams
scripts/
  severing_test.py      bottleneck-severing test (inference only): replaces the four-value code with the
                        mean training code (const) or a random training image's code (swap)
tests/                  unit tests (data preparation, circuit vs. reference, gradients, statistics;
                        test_base_ansatz.py: Jacobian ranks 20/24 and 88/96)
```

## Usage

```bash
pip install -r requirements.txt
```

```python
from qseg.config import Config
from qseg.data_prep import prepare_dataset
from qseg.train import run_all

cfg = Config(raw_image_dir="path/to/images", work_dir="results")
prepare_dataset(cfg)          # proxy masks + patient-level split (split seed 42)
run_all(cfg)                  # 5 models x seeds 0, 1, 2
```

Statistics reported in the paper (10,000 bootstrap replicates):

```bash
python -m qseg.final_stats results results/final_stats_output.txt
```

Tests:

```bash
python -m pytest -q tests
```

`tests/test_tf_circuit.py` checks that the TensorFlow circuit matches the NumPy reference and that
exactly four circuit parameters (the final-layer R_z(ω) rotations) are inert, so the circuit has
20 effective parameters.

## Bottleneck-severing test

`scripts/severing_test.py` loads the nine trained hybrid runs, replaces the four-value bottleneck code
at inference time (const: mean code of the training images; swap: code of a random training image)
and reports the drop in per-image Dice with a hierarchical bootstrap over patients and seeds and
Holm correction. It first checks that the network split at the code reproduces the full network and
the saved per-image Dice. Written for Google Colab (see the paths at the top of the script).

## Check of the base-design ansatz

The circuit follows the base design of S.-C. Lim and J.-C. Kim, "Hybrid quantum-classical UNet for image
segmentation: Breaking the parameter scaling barrier," *IEEE Access*, vol. 14, pp. 33935-33947, 2026,
doi: [10.1109/ACCESS.2026.3669206](https://doi.org/10.1109/ACCESS.2026.3669206).
`qseg/base_ansatz_check.py` (NumPy only, a few seconds) simulates both circuits exactly and stacks
the central finite-difference Jacobians d<Z_i>/dw over random inputs (uniform in [-1, 1]) and random
parameter settings (uniform in [0, 2π), fixed seeds): our circuit (4 qubits, 2 layers, R_y(π/2·x)
encoding; 64 inputs × 3 settings) and Lim & Kim's ansatz as specified in their eqs. (3)-(8)
(8 qubits, 4 layers, R_y(x) encoding, Rot(θ, φ, λ) = R_z(φ) R_y(θ) R_z(λ) and a CNOT ring;
24 inputs × 2 settings).

```bash
python -m qseg.base_ansatz_check
```

Expected output: rank **20 of 24** for our circuit (zero columns: the four final-layer R_z(ω)) and
rank **88 of 96** for Lim & Kim's ansatz (zero columns: the eight final-layer R_z(φ), the rotations
applied last, directly before the Z measurement). `tests/test_base_ansatz.py` asserts these values.

## Environment used for the reported runs

Python 3.13, TensorFlow/Keras 2.20.0, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, Google Colab with an
NVIDIA Tesla T4 GPU. The quantum backend for all reported runs is `"tf"`; the optional PennyLane
layer in `models.py` was not used.

## Citation

S. Parii, "hqb-unet-ablation: Source code for the parameter-matched ablation of a quantum-bottleneck U-Net (v1.0)," Zenodo, 2026, doi: 10.5281/zenodo.23390486.

## License

MIT (see `LICENSE`).

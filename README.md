# DySIB Pendulum

Code and data for the paper:

> **DySIB: Learning True Dynamical Coordinates from High-Dimensional Observations**
> K. Michael Martini, Eslam Abdelaleem, Paarth Gulati, and Ilya Nemenman. *[preprint]*, 2026.
> [DOI / arXiv link]

---

## What this repository does

**DySIB** (**Dy**namical **S**ymmetric **I**nformation **B**ottleneck) is a method that learns a compressed, low-dimensional representation of a dynamical system's state directly from high-dimensional observations — without labels or prior knowledge of the physics. Given a sequence of frames showing the system evolving over time, DySIB finds the coordinates that carry maximal information about the future, and those coordinates turn out to correspond to the true physical variables of the system.

This repository demonstrates DySIB on a rigid pendulum: the input is a video of the pendulum (128×128 RGB frames), and the method discovers 2 latent dimensions that align with angle and angular velocity. The same training code can be applied to other dynamical systems by providing trajectory data in the same format (see [DETAILS.md](DETAILS.md) for how to adapt the data pipeline).

The repository provides:
- The full training code and pretrained models ready to use
- The pendulum video dataset (downloaded automatically on first use)
- A single notebook that regenerates every figure in the paper

---

## Quickstart: reproduce the paper figures (no training needed)

```bash
git clone https://github.com/paarthgulati/DYSIB_Pendulum.git
cd DYSIB_Pendulum
pip install -r requirements.txt
jupyter lab reproduce_figures.ipynb
```

The notebook loads pretrained models and precomputed data files that ship with the repository. It runs in under a minute on any laptop — no GPU, no downloading data. Figures are saved to `Figures/`.

---

## Installation

Requires **Python ≥ 3.9**.

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended for training but is not needed to reproduce figures.

---

## Running the full pipeline from scratch

### Step 1 — Download and preprocess the data

```bash
python data.py --all
```

This fetches the [Chen et al. (2022)](https://zenodo.org/records/6653856) pendulum video dataset from Zenodo (~167 MB), extracts 1200 trajectories (60 frames each at 128×128 RGB), downsamples each frame to 28×28 grayscale, and saves `data/pendulum_preprocessed.npz` (~227 MB).

### Step 2 — Train a model

```bash
python training.py \
    --kz 2 --num_frames 2 --samples_n 1000 \
    --rep 0 --epochs 300 \
    --out checkpoints/my_run.pt
```

Key options:

| Flag | Meaning | Paper default |
|------|---------|--------------|
| `--kz` | Latent dimension | 2 |
| `--num_frames` | Frames per input window | 2 |
| `--samples_n` | Training trajectories (max 1000) | 1000 |
| `--rep` | Random seed | varies |
| `--epochs` | Training epochs | 300 |
| `--gamma` | KL compression weight γ; together with `--alpha`, sets β = α/γ | 0.01 |
| `--alpha` | MI maximisation weight α | 1.0 |

Training prints mutual information (MI) and loss per epoch. On CPU, 300 epochs takes roughly 20–30 minutes; on a GPU it is much faster.

### Step 3 — Regenerate analysis files

```bash
python precompute_figure_data.py
```

This loads the checkpoints in `checkpoints/`, encodes the dataset, trains linear probers, and writes the precomputed arrays consumed by the notebook to `figure_data/`. To use your own trained checkpoint instead of the shipped ones, update the `CKPT_KZ2` / `CKPT_KZ8` paths at the top of `precompute_figure_data.py`.

### Step 4 — Plot the figures

```bash
jupyter lab reproduce_figures.ipynb
```

---

## Repository layout

```
DYSIB_Pendulum/
├── models.py                   DYSIB, Encoder, DeltaDecoder, Prober, FrameDataset
├── training.py                 Training loop + prober training + encoding + rollout + CLI
├── data.py                     Download + extract + preprocess + load + CLI
├── precompute_figure_data.py   Compute figure_data/ from checkpoints + data
├── reproduce_figures.ipynb     Generate all paper figures from figure_data/
├── requirements.txt
├── LICENSE                     MIT
│
├── checkpoints/                Trained model weights (shipped, ~4.5 MB total)
│   ├── kz2_nF2_N1000_rep4.pt   Main embedding figures (kz=2, 1000 training trials)
│   ├── kz8_nF2_N1000_rep21.pt  Intrinsic-dimensionality figure (kz=8)
│   ├── kz2_nF2_N64_rep4.pt     
│   └── kz2_nF2_N256_rep4.pt    Scalability comparison (N = 64, 256)
│
├── figure_data/                Precomputed arrays for the notebook (shipped, ~7 MB)
│   ├── sweep_metrics.csv       Metrics from 571-run hyperparameter sweep
│   ├── phase_portrait.npz
│   ├── embedding_kz2.npz
│   ├── embedding_vs_N.npz
│   ├── rollout_test.npz
│   ├── pca_kz8.npz
│   └── cylindrical_phys.npz
│
├── data/                       Local data cache (git-ignored, created on first run)
│   ├── single_pendulum.zip     Raw archive from Zenodo
│   ├── single_pendulum/        Extracted frames (1200 trajectories × 60 PNG files)
│   └── pendulum_preprocessed.npz  Preprocessed frames + physical labels (~227 MB)
│
└── Figures/                    Output directory for saved figures (git-ignored)
```

---

## Using the code programmatically

Every function used in the pipeline is importable directly:

```python
import torch
from models   import DYSIB, Prober, FrameDataset
from training import train_dysib, train_prober, encode_dataset, rollout_latent
import data as data_mod

# Load data
d = data_mod.load()                       # downloads + preprocesses on first call
frames = torch.from_numpy(d["frames"])    # (1200, 60, 784)

# Build and train a model
model = DYSIB(input_dim=784, dz=2)
model, history = train_dysib(model, loader, x_test, y_test, epochs=300)

# Encode the dataset and roll out latent dynamics
zx, numD = encode_dataset(model, frames, num_frames=2)
rollout   = rollout_latent(model, zx, num_frames=2, numD=numD)
```

See [DETAILS.md](DETAILS.md) for full function signatures, architecture details, checkpoint format, and notes on extending the code.

---

## Citation

If you use this code or the trained models, please cite:

```bibtex
@article{martini2026dysib,
  title   = {DySIB: Learning True Dynamical Coordinates from High-Dimensional Observations},
  author  = {Martini, K. Michael and Abdelaleem, Eslam and Gulati, Paarth and Nemenman, Ilya},
  journal = {[preprint]},
  year    = {2026},
}
```

The raw video dataset is from:

```bibtex
@article{chen2022discovering,
  title   = {Automated discovery of fundamental variables hidden in experimental data},
  author  = {Chen, Boyuan and Huang, Kuang and Raghupathi, Sunand and
             Chandratreya, Ishaan and Du, Qiang and Lipson, Hod},
  journal = {Nature Computational Science},
  year    = {2022},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

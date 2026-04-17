# DYSIB Pendulum

Code and data for the paper:

> **[Paper title here]**
> [Author list]. *[Journal / preprint]*, 2026.
> [DOI / arXiv link]

DYSIB (Dynamical Symmetric Information Bottleneck) is a representation-learning
method that extracts low-dimensional latent descriptions of dynamical systems
from high-dimensional observations. This repository trains DYSIB on video
recordings of a rigid pendulum and reproduces every figure in the paper.

---

## Installation

```bash
git clone https://github.com/<user>/DYSIB_Pendulum.git
cd DYSIB_Pendulum
pip install -r requirements.txt
```

Requires Python ≥ 3.9. A CUDA-capable GPU is recommended for training but not
required for figure reproduction.

---

## Data

The raw pendulum video dataset is from
[Chen et al. (2022)](https://zenodo.org/records/6653856) — specifically the
`single_pendulum.zip` file (167 MB, 1200 trajectories × 60 frames of 128×128 RGB).
On first use, download and preprocess:

```bash
python data.py --all
```

This (a) downloads `single_pendulum.zip` from Zenodo, (b) extracts it into
`data/single_pendulum/`, and (c) downsamples each frame to **28×28 grayscale**,
normalized to [0, 1], saved as `data/pendulum_preprocessed.npz` (~226 MB).
Subsequent runs read the cached `.npz`.

To change the preprocessing (e.g. different resolution, interpolation, or
normalization) without editing code:

```bash
python data.py --preprocess --size 32 --downsampler lanczos --normalizer meanstd --force
```

Available choices: downsamplers = `bilinear | bicubic | lanczos | nearest`;
normalizers = `div255 | meanstd | minmax | identity`. Register additional
methods in `DOWNSAMPLERS` / `NORMALIZERS` dicts in [data.py](data.py).

---

## Reproduce the paper figures

No training required. The repository ships four trained checkpoints
(`checkpoints/*.pt`, ~4.5 MB total) for runs that appear in embedding figures
and precomputed analysis arrays (`figure_data/*.npz`, `figure_data/sweep_metrics.csv`,
~7 MB total) for every figure.

```bash
jupyter lab reproduce_figures.ipynb
```

Runs end-to-end in under a minute using only `numpy`, `pandas`, `scipy`,
`matplotlib`, and `cmocean` — **no torch, no GPU**.

Figures are saved under `Figures/` at publication DPI (`SAVE_DPI` at the top of
the notebook; default 600, increase to 1000 for final print).

### Rebuilding the precomputed data

If you retrain models or change the preprocessing, regenerate the analysis arrays:

```bash
python precompute_figure_data.py               # regenerate everything
python precompute_figure_data.py --only rollout_test --force   # single task
```

Available tasks: `sweep_metrics`, `phase_portrait`, `embedding_kz2`,
`embedding_vs_N`, `rollout_test`, `pca_kz8`, `cylindrical_phys`.

---

## Train from scratch

Train a single DYSIB model with custom parameters:

```bash
python training.py --kz 2 --num_frames 2 --samples_n 1000 --rep 0 \
                   --epochs 300 --out checkpoints/my_run.pt
```

Key parameters:
- `--kz`         latent dimension
- `--num_frames` frames per window
- `--samples_n`  number of training trials (1–1000)
- `--rep`        random seed / repetition index
- `--gamma`      KL weight (default 0.01)

### End-to-end pipeline

```
Raw data  ─ python data.py --all ─►  data/pendulum_preprocessed.npz
           │                        (local cache, ~226 MB, git-ignored)
           │
           ├─ python training.py --out checkpoints/<name>.pt ─►  checkpoints/<name>.pt
           │                                                     (~1.1 MB per run)
           │
           └─ python precompute_figure_data.py ─►  figure_data/*.npz + .csv
                                                   (~7 MB, git-tracked)
                                                           │
                                jupyter lab reproduce_figures.ipynb
                                                           ↓
                                                      Figures/*.png
```

Each `.pt` checkpoint is a single dict with keys `state_dict`, `params`, `history`
(saved by `torch.save`, ~1.1 MB). See the `_run_training` docstring in
[training.py](training.py) for the exact schema.

### Plugging your own checkpoints into the figure pipeline

`precompute_figure_data.py` loads checkpoints by the hard-coded filenames at
the top of the script:

```python
CKPT_KZ2       = checkpoints/kz2_nF2_N1000_rep4.pt
CKPT_KZ8       = checkpoints/kz8_nF2_N1000_rep21.pt
CKPT_KZ2_N64   = checkpoints/kz2_nF2_N64_rep4.pt
CKPT_KZ2_N256  = checkpoints/kz2_nF2_N256_rep4.pt
```

To substitute your own run, either save your `.pt` to the matching path, or
edit those constants. The `sweep_metrics.csv` file is shipped pre-aggregated
from a 571-run sweep and cannot be regenerated from a single training run —
edit `task_sweep_metrics` in [precompute_figure_data.py](precompute_figure_data.py)
if you have your own sweep to aggregate (it reads the legacy HDF5 layout
described at the top of that function).

### Using the code without the precompute script

Every piece of analysis the notebook shows is a thin wrapper over library
functions exposed by `training.py` and `models.py`. If you want a different
workflow — e.g. train and plot in one notebook, export to a different format,
or use your own aggregation — import what you need directly:

```python
from models   import DYSIB, Prober, FrameDataset
from training import train_dysib, train_prober, encode_dataset, rollout_latent
```

See the module-level docstrings in [training.py](training.py) and [models.py](models.py)
for the full function signatures.

---

## Repository layout

```
DYSIB_Pendulum/
├── models.py                   DYSIB, Encoder, DeltaDecoder, Prober, FrameDataset
├── training.py                 Training + prober + encoding + rollout + CLI
├── data.py                     Download + extract + preprocess + load + CLI
├── precompute_figure_data.py   Generate figure_data/ from checkpoints/ + data/
├── reproduce_figures.ipynb     Consolidated plotting notebook
├── requirements.txt
├── LICENSE                     MIT
├── checkpoints/                .pt weights for runs used in embedding figures
│   ├── kz2_nF2_N1000_rep4.pt
│   ├── kz2_nF2_N64_rep4.pt
│   ├── kz2_nF2_N256_rep4.pt
│   └── kz8_nF2_N1000_rep21.pt
├── figure_data/                Precomputed arrays consumed by the notebook
│   ├── sweep_metrics.csv       MI + prober RMSE across all sweep runs
│   ├── phase_portrait.npz
│   ├── embedding_kz2.npz
│   ├── embedding_vs_N.npz
│   ├── rollout_test.npz
│   ├── pca_kz8.npz
│   └── cylindrical_phys.npz
├── data/                       (local cache, git-ignored)
│   ├── single_pendulum.zip
│   ├── single_pendulum/        extracted raw frames (1200 trajectories)
│   └── pendulum_preprocessed.npz
└── Figures/                    (output dir for savefig, git-ignored)
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{todo,
  title   = {},
  author  = {},
  journal = {},
  year    = {2026},
}
```

The raw video dataset is from:

```bibtex
@article{chen2022discovering,
  title   = {Automated discovery of fundamental variables hidden in experimental data},
  author  = {Chen, Boyuan and Huang, Kuang and Raghupathi, Sunand and Chandratreya, Ishaan and Du, Qiang and Lipson, Hod},
  journal = {Nature Computational Science},
  year    = {2022},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

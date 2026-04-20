# Technical Details

This document covers the model architecture, training objective, data pipeline internals, checkpoint format, and notes for people who want to extend or adapt the code. The paper writes the method as "DySIB"; the code class is `DYSIB` — same method, different capitalisation.

---

## Model architecture

The full model (`DYSIB` in `models.py`) has three components:

### Encoder

A per-frame MLP that maps a sliding window of frames to a VAE latent code.

```
Input:  (batch, num_frames, D)    — D = 784 for 28×28 frames
                ↓
        3-layer MLP applied independently to each frame
        [Linear(D, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, dzf)]
                ↓
        Concatenate frame embeddings → (batch, num_frames × dzf)
                ↓
        Two linear heads: mean_z, logvar_z  (batch, dz)
                ↓
Output: z ~ N(mean_z, exp(logvar_z))  (batch, dz)
```

Default: `dz=2` (k_z in the paper), `dzf=32` (d_F in the paper), `num_frames=2` (n_F in the paper), `hidden=256`.

### DeltaDecoder (δ-predictor in the paper)

A small MLP that predicts the *change* in latent state from one time step to the next.

```
Input:  zx  (batch, dz)
Output: mean_delta, logvar  (batch, dz)

Prediction: z_future ≈ zx + mean_delta          (greedy)
            z_future ~ N(zx + mean_delta, exp(logvar))  (stochastic)
```

Default: `hdim=64`, 3 hidden layers.

### Prober

A small MLP trained on top of frozen DYSIB latent codes to decode physical variables (angle θ, angular velocity ω, energies). Trained separately after DYSIB training is complete.

```
Input:  z  (batch, dz)
Output: predicted physical variable  (batch, output_dim)
```

Default: `hdim=1024`, 3 hidden layers.

---

## Training objective

The DySIB objective is:

```
loss = γ · (KL_x + KL_y) − α · I_NCE
```

`γ` (`--gamma`) and `α` (`--alpha`) are independent weights on the compression and MI terms. The paper fixes `γ = 0.01`, `α = 1.0`, which is equivalent to the paper's β = α / γ = 100. Changing either weight independently lets you explore other points on the compression-prediction curve.

**Mutual information term** uses Gaussian InfoNCE (contrastive predictive coding). The DeltaDecoder (δ-predictor) predicts a Gaussian distribution over the future latent code, and the InfoNCE score for a batch of size B is:

```
scores[i, j] = log N(zy_j ; zx_i + delta_i, exp(logvar_i))
I_NCE = mean_i [ scores[i,i] − logsumexp_j(scores[i,j]) + log B ]
```

**KL term** is the standard VAE regularizer encouraging the posterior to stay close to a unit Gaussian.

**Symmetry**: the same encoder is used for both the past window `X` and the future window `Y`. Both KL terms (from encoding past and future) are summed.

---

## Data pipeline

The dataset is sourced from [Chen et al. 2022](https://zenodo.org/records/6653856), which contains 1200 trajectories of a rigid pendulum simulated with known physical parameters. Each trajectory has 60 frames at 128×128 RGB.

**Preprocessing** (`data.py`):
1. Convert RGB → grayscale using luminance weighting
2. Resize 128×128 → 28×28 (default: bilinear interpolation)
3. Normalize pixel values to [0, 1] (default: divide by 255)
4. Store as float32 alongside physical ground truth (θ, ω, KE, PE, TE)

**Train/test split** (hardcoded in `training.py`):
- Training: trials 0 through `samples_n − 1` (max 1000)
- Test: trials 1000–1199 (always fixed, 200 trajectories)

The split is fixed so that evaluation is always on held-out data. Passing `--samples_n > 1000` raises an error because it would overlap the test split.

**Sliding window pairs** (`FrameDataset`):
For a trajectory of length T with window size `num_frames`, there are `numD = T − 2·num_frames + 1` valid start positions. Each item is a pair (past_window, future_window) where the future window immediately follows the past window.

---

## Checkpoint format

Each `.pt` file is a plain Python dict saved with `torch.save`:

```python
{
    "state_dict": { ... },   # model weights (Encoder + DeltaDecoder)
    "params": {              # hyperparameters used during training
        "kz":           int,    # latent dimension
        "num_frames":   int,
        "samples_n":    int,
        "rep":          int,    # random seed
        "dzf":          int,
        "hidden":       int,
        "decoder_hdim": int,
        "gamma":        float,
        "alpha":        float,
        "input_dim":    int,    # D (pixels per frame)
    },
    "history": {             # per-epoch training curves
        "train_loss": [...],
        "train_kl":   [...],
        "train_mi":   [...],
        "test_mi":    [...],
    }
}
```

Load with:

```python
import torch
ck = torch.load("checkpoints/kz2_nF2_N1000_rep4.pt", weights_only=False)
model.load_state_dict(ck["state_dict"])
```

The four shipped checkpoints were saved with an older training script and use `"ndims"` instead of `"kz"` as the latent dimension key. The `_load_checkpoint` helper in `precompute_figure_data.py` normalises this automatically.

---

## Precomputed figure data

`precompute_figure_data.py` generates the files in `figure_data/` from the shipped checkpoints. Running it requires the full dataset (`python data.py --all`) and takes several minutes on CPU.

| File | Contents |
|------|----------|
| `sweep_metrics.csv` | Flat table of metrics across 571 hyperparameter runs: kz, num_frames, N, rep, final MI, prober RMSE for θ/ω/energy |
| `phase_portrait.npz` | Latent codes + wrapped angle + angular velocity for kz=2 model |
| `embedding_kz2.npz` | Subsampled (100k) latent codes + physical variables, for scatter plots |
| `embedding_vs_N.npz` | Latent codes from kz=2 models trained with N=64, 256, 1000 |
| `rollout_test.npz` | Rollout RMSE curves, quiver field, fixed points, visualization trajectories |
| `pca_kz8.npz` | Singular values, participation ratio, TwoNN/MLE intrinsic dimensionality, PCA projection |
| `cylindrical_phys.npz` | Wrapped angle (degrees) and angular velocity (degrees/s) for the cylindrical figure |

`sweep_metrics.csv` is pre-aggregated from a 571-run sweep stored in legacy HDF5 files. It is not regeneratable from a single training run; it ships as-is and is read directly by the notebook.

To use your own trained checkpoints with the precompute script, update the `CKPT_KZ2` / `CKPT_KZ8` / `CKPT_KZ2_N64` / `CKPT_KZ2_N256` paths at the top of `precompute_figure_data.py`. You can also run individual tasks:

```bash
python precompute_figure_data.py --only embedding_kz2 rollout_test
python precompute_figure_data.py --force   # overwrite existing files
```

---

## Key function signatures

### `train_dysib(model, loader, x_test, y_test, epochs, lr, device)`

Train a DYSIB model. Returns `(model, history_dict)`.

- `loader` yields `(past_frames, future_frames)` batches of shape `(B, num_frames, D)`
- `x_test`, `y_test` are fixed test batches already on device, used for per-epoch MI evaluation
- Includes gradient clipping (norm 1.0) and NaN detection

### `encode_dataset(model, data, num_frames, device, batch_size)`

Encode all sliding windows and return posterior means (not samples).

- `data`: `(n_trials, T, D)` float tensor
- Returns `(zx, numD)` where `zx` has shape `(n_trials × numD, dz)`, ordered by (trial, window) with trial as the slow index

### `rollout_latent(model, zx, num_frames, numD, device, num_repeats, extra_steps)`

Roll out the learned latent dynamics from the first window of each trajectory.

- Returns a dict with greedy path, stochastic mean/std path, and latent-space RMSE curves

### `train_prober(prober, z_train, y_train, z_test, y_test, epochs, lr, batch_size, device)`

Train a regression probe on frozen latent codes. Returns `(prober, train_mse, test_mse)`.

---

## Intrinsic dimensionality estimators

The kz=8 analysis uses two estimators from the `skdim` package:

- **TwoNN** (`skdim.id.TwoNN`): estimates intrinsic dimension from the ratio of 1st and 2nd nearest-neighbour distances. Computationally cheap and parameter-free.
- **MLE** (`sklearn.neighbors.NearestNeighbors`): Levina–Bickel maximum likelihood estimator. Computed across a range of k values to show stability.

Both are run on a 10,000-point subsample of the full latent dataset.

---

## Extending the code

**Different dataset**: `DYSIB`, `FrameDataset`, and `train_dysib` are dataset-agnostic — they operate on any `(n_trials, T, D)` float tensor of frame observations. To apply DySIB to a new system, prepare your trajectory data in that format and pass it directly to `FrameDataset` and `train_dysib`. The `data.py` module is specific to the pendulum Zenodo dataset and does not need to be used.

**Different resolution**: pass `--size 32` to `data.py --preprocess` and then `--data` pointing to the new `.npz` in `training.py`. The model infers `input_dim` from the data.

**Different interpolation / normalization**: `data.py` has pluggable `DOWNSAMPLERS` and `NORMALIZERS` dicts. Register a new function and pass its key via `--downsampler` / `--normalizer`.

**Deeper encoder / decoder**: the `hidden` and `decoder_hdim` arguments control hidden layer widths. The number of layers is fixed at 3 in both components.

**Different latent dimension**: just change `--kz`. For `kz=1` the model should converge to a 1D representation correlated with angle; for `kz ≥ 3` you can use the kz=8 analysis pipeline to check whether extra dimensions are used.

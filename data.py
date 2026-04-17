"""
Download, extract, preprocess, and load the single-pendulum dataset.

Raw data source (Zenodo):
    https://zenodo.org/records/6653856/files/single_pendulum.zip
    (supports Chen et al. "Discovering State Variables Hidden in Experimental Data".)

Raw archive layout:
    single_pendulum/
    ├── phys_vars.npy                   # dict: theta, vel_theta, energies, reject
    ├── states.npy                      # (unused here)
    ├── 0/   {0..59}.png   128x128 RGB
    ├── 1/   {0..59}.png
    ...
    └── 1199/

Preprocessed output (local cache, not committed):
    data/pendulum_preprocessed.npz
        frames           (1200, 60, D)  float32   — D = size*size downsampled grayscale
        theta            (1200, 60)     float32
        vel_theta        (1200, 60)     float32
        kinetic_energy   (1200, 60)     float32
        potential_energy (1200, 60)     float32
        total_energy     (1200, 60)     float32
        reject           (1200, 60)     bool

─ Pluggable preprocessing ────────────────────────────────────────────────────

Frame preprocessing is factored into two functions registered in dictionaries:

    DOWNSAMPLERS[name](img_rgb_128) -> np.ndarray shape (size, size) float32
    NORMALIZERS[name](arr)           -> np.ndarray, normalized in place style

Pick at call time (CLI: --downsampler / --normalizer, or via preprocess(...)).
To add a new method, register a function in the corresponding dict below.

Default: bilinear downsample on luminance grayscale, divide by 255.
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── constants ────────────────────────────────────────────────────────────────
ZENODO_URL   = "https://zenodo.org/records/6653856/files/single_pendulum.zip?download=1"
DEFAULT_ROOT = Path("data")
ZIP_NAME     = "single_pendulum.zip"
RAW_SUBDIR   = "single_pendulum"
NPZ_NAME     = "pendulum_preprocessed.npz"

N_TRIALS = 1200
N_FRAMES = 60


# ─── downsamplers (RGB 128×128 → grayscale size×size, float32) ──────────────
def _ds_bilinear(img_rgb, size):
    return np.asarray(
        Image.fromarray(img_rgb).convert("L").resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    )

def _ds_bicubic(img_rgb, size):
    return np.asarray(
        Image.fromarray(img_rgb).convert("L").resize((size, size), Image.BICUBIC),
        dtype=np.float32,
    )

def _ds_lanczos(img_rgb, size):
    return np.asarray(
        Image.fromarray(img_rgb).convert("L").resize((size, size), Image.LANCZOS),
        dtype=np.float32,
    )

def _ds_nearest(img_rgb, size):
    return np.asarray(
        Image.fromarray(img_rgb).convert("L").resize((size, size), Image.NEAREST),
        dtype=np.float32,
    )

DOWNSAMPLERS = {
    "bilinear": _ds_bilinear,
    "bicubic":  _ds_bicubic,
    "lanczos":  _ds_lanczos,
    "nearest":  _ds_nearest,
}


# ─── normalizers (array in, array out; operates on float32 in [0, 255]) ─────
def _nm_div255(arr):
    """Standard [0, 1] normalization by dividing by 255 (option A)."""
    return arr / 255.0

def _nm_meanstd(arr):
    """Zero-mean unit-variance over the full dataset (global)."""
    m, s = arr.mean(), arr.std()
    return (arr - m) / (s if s > 0 else 1.0)

def _nm_minmax(arr):
    """Min-max scale to [0, 1] using per-dataset extremes."""
    a, b = arr.min(), arr.max()
    return (arr - a) / (b - a if b > a else 1.0)

def _nm_identity(arr):
    """No normalization — values remain in [0, 255]."""
    return arr

NORMALIZERS = {
    "div255":   _nm_div255,
    "meanstd":  _nm_meanstd,
    "minmax":   _nm_minmax,
    "identity": _nm_identity,
}


# ─── download ────────────────────────────────────────────────────────────────
def download(root=DEFAULT_ROOT, force=False):
    """Fetch single_pendulum.zip from Zenodo into `root`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / ZIP_NAME
    if zip_path.exists() and not force:
        print(f"[data] Already downloaded: {zip_path}")
        return zip_path

    # Stream with progress bar; urllib is stdlib (no extra dep).
    import urllib.request
    print(f"[data] Downloading {ZENODO_URL}")
    with urllib.request.urlopen(ZENODO_URL) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="download"
        ) as bar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"[data] Saved: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


# ─── extract ────────────────────────────────────────────────────────────────
def extract(zip_path, root=DEFAULT_ROOT, force=False):
    """Unzip the archive into `root/single_pendulum/`."""
    root = Path(root)
    raw_dir = root / RAW_SUBDIR
    sentinel = raw_dir / "phys_vars.npy"
    if sentinel.exists() and not force:
        print(f"[data] Already extracted: {raw_dir}")
        return raw_dir

    print(f"[data] Extracting {zip_path} → {root}/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not sentinel.exists():
        raise RuntimeError(
            f"Extraction completed but {sentinel} not found — check archive layout."
        )
    return raw_dir


# ─── preprocess ─────────────────────────────────────────────────────────────
def preprocess(
    raw_dir,
    out_path,
    size=28,
    downsampler="bilinear",
    normalizer="div255",
    force=False,
):
    """Downsample frames, bundle with phys vars, save a single .npz.

    Args:
        raw_dir       : extracted single_pendulum/ directory
        out_path      : .npz file to write
        size          : downsampled spatial resolution (28 → 28×28 = 784 pixels)
        downsampler   : key in DOWNSAMPLERS
        normalizer    : key in NORMALIZERS
        force         : overwrite existing out_path

    Returns: out_path
    """
    raw_dir  = Path(raw_dir)
    out_path = Path(out_path)
    if out_path.exists() and not force:
        print(f"[data] Already preprocessed: {out_path}")
        return out_path

    if downsampler not in DOWNSAMPLERS:
        raise KeyError(f"Unknown downsampler {downsampler!r}; choices: {list(DOWNSAMPLERS)}")
    if normalizer not in NORMALIZERS:
        raise KeyError(f"Unknown normalizer  {normalizer!r}; choices: {list(NORMALIZERS)}")

    ds_fn = DOWNSAMPLERS[downsampler]
    nm_fn = NORMALIZERS[normalizer]

    D = size * size
    frames = np.empty((N_TRIALS, N_FRAMES, D), dtype=np.float32)

    print(f"[data] Downsampling {N_TRIALS}×{N_FRAMES} frames "
          f"(128×128 RGB → {size}×{size} grayscale, method={downsampler})")
    for trial in tqdm(range(N_TRIALS), desc="trials"):
        tdir = raw_dir / str(trial)
        for t in range(N_FRAMES):
            img_rgb = np.asarray(Image.open(tdir / f"{t}.png").convert("RGB"))
            gray    = ds_fn(img_rgb, size)       # (size, size) float32 in [0, 255]
            frames[trial, t] = gray.ravel()

    # Apply normalization globally (so any scheme that depends on dataset stats
    # sees the full distribution).
    frames = nm_fn(frames).astype(np.float32, copy=False)
    print(f"[data] frames: shape={frames.shape}, "
          f"dtype={frames.dtype}, min={frames.min():.3f}, max={frames.max():.3f}")

    # Physical variables (already in the correct shape and format).
    phys = np.load(raw_dir / "phys_vars.npy", allow_pickle=True).item()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        frames           = frames,
        theta            = phys["theta"].astype(np.float32),
        vel_theta        = phys["vel_theta"].astype(np.float32),
        kinetic_energy   = phys["kinetic energy"].astype(np.float32),
        potential_energy = phys["potential energy"].astype(np.float32),
        total_energy     = phys["total energy"].astype(np.float32),
        reject           = phys["reject"],
        meta_downsampler = np.array(downsampler),
        meta_normalizer  = np.array(normalizer),
        meta_size        = np.array(size),
    )
    print(f"[data] Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ─── load ───────────────────────────────────────────────────────────────────
def load(npz_path=None, root=DEFAULT_ROOT, auto=True, **preprocess_kwargs):
    """Load preprocessed data; if missing and auto=True, run the full pipeline.

    Returns a dict with keys:
        frames, theta, vel_theta, kinetic_energy, potential_energy,
        total_energy, reject
    (plus meta_* informational keys).
    """
    root = Path(root)
    npz_path = Path(npz_path) if npz_path is not None else root / NPZ_NAME

    if not npz_path.exists():
        if not auto:
            raise FileNotFoundError(f"{npz_path} not found and auto=False")
        zip_path = download(root=root)
        raw_dir  = extract(zip_path, root=root)
        preprocess(raw_dir, npz_path, **preprocess_kwargs)

    arrs = np.load(npz_path, allow_pickle=False)
    return {k: arrs[k] for k in arrs.files}


# ─── CLI ────────────────────────────────────────────────────────────────────
def _main():
    p = argparse.ArgumentParser(description="Download + preprocess the pendulum dataset.")
    p.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                   help="Directory for downloads and cached outputs (default: data/)")
    p.add_argument("--download",   action="store_true", help="Download the archive")
    p.add_argument("--extract",    action="store_true", help="Extract the archive")
    p.add_argument("--preprocess", action="store_true", help="Downsample + bundle npz")
    p.add_argument("--all",        action="store_true", help="Run all three steps")
    p.add_argument("--force",      action="store_true", help="Overwrite cached outputs")
    p.add_argument("--size", type=int, default=28,
                   help="Downsampled spatial resolution (default: 28)")
    p.add_argument("--downsampler", choices=list(DOWNSAMPLERS), default="bilinear")
    p.add_argument("--normalizer",  choices=list(NORMALIZERS),  default="div255")
    args = p.parse_args()

    root = Path(args.root)

    if args.all or args.download:
        zip_path = download(root=root, force=args.force)
    else:
        zip_path = root / ZIP_NAME

    if args.all or args.extract:
        raw_dir = extract(zip_path, root=root, force=args.force)
    else:
        raw_dir = root / RAW_SUBDIR

    if args.all or args.preprocess:
        preprocess(
            raw_dir, root / NPZ_NAME,
            size=args.size,
            downsampler=args.downsampler,
            normalizer=args.normalizer,
            force=args.force,
        )


if __name__ == "__main__":
    _main()

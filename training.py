"""
Training and evaluation functions for DYSIB on pendulum dynamics.

Functions:
    train_dysib                  -- train a DYSIB model, return (model, history dict)
    train_prober                 -- train a Prober with MSE loss, return (prober, train_mse, test_mse)
    predict_prober               -- run a trained prober in mini-batches
    wrapped_theta_rmse_deg_from_cos_sin
                                 -- wrapped angular RMSE from [cos(theta), sin(theta)] predictions
    encode_dataset               -- encode all (trial, window) pairs, return latent codes
    rollout_latent               -- greedy + stochastic rollout in latent space
    rollout_rms                  -- physical-variable RMS along a rollout vs ground truth
    rollout_theta_rmse_wrapped   -- wrapped angular RMSE along a rollout for theta
"""

import numpy as np
import torch
from tqdm import tqdm

from models import FrameDataset, ProbeDataset, Prober


# ─────────────────────────────────────────────────────────────────────────────
# DYSIB training
# ─────────────────────────────────────────────────────────────────────────────

def train_dysib(model, loader, x_test, y_test, epochs, lr=1e-4, device="cuda"):
    """
    Train a DYSIB model.

    Args:
        model         : DYSIB instance (already on device)
        loader        : DataLoader yielding (past_frames, future_frames) batches
        x_test, y_test: test batches (already on device) for per-epoch eval
        epochs        : number of training epochs
        lr            : Adam learning rate
        device        : torch device string

    Returns:
        model : trained model
        hist  : dict with per-epoch lists
                  train_loss, train_kl, train_mi, test_mi
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "train_kl": [], "train_mi": [], "test_mi": []}

    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        run_loss = run_kl = run_mi = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss, kl, mi_term, mi = model(x, y)

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch + 1}; stopping.")
                return model, hist

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += loss.item()
            run_kl   += kl.item()
            run_mi   += mi.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            _, _, _, test_mi = model(x_test, y_test)

        tr_loss = run_loss / n_batches
        tr_kl   = run_kl   / n_batches
        tr_mi   = run_mi   / n_batches
        ts_mi   = test_mi.item()

        hist["train_loss"].append(tr_loss)
        hist["train_kl"].append(tr_kl)
        hist["train_mi"].append(tr_mi)
        hist["test_mi"].append(ts_mi)

        pbar.set_postfix({
            "loss":  f"{tr_loss:.3f}",
            "MI_tr": f"{tr_mi:.3f}",
            "MI_ts": f"{ts_mi:.3f}",
        })

    return model, hist


# ─────────────────────────────────────────────────────────────────────────────
# Prober training
# ─────────────────────────────────────────────────────────────────────────────

def train_prober(prober, z_train, y_train, z_test, y_test,
                 epochs=100, lr=1e-4, batch_size=1024, device="cuda"):
    """
    Train a Prober to predict a physical variable from latent codes.
    The DYSIB model is NOT touched here; z_train / z_test are precomputed.

    Args:
        prober          : Prober instance (already on device)
        z_train, y_train: CPU tensors, shapes (N_tr, dz) and (N_tr, Dy)
        z_test,  y_test : CPU tensors, shapes (N_ts, dz) and (N_ts, Dy)
        epochs, lr, batch_size, device: standard training hyperparameters

    Returns:
        prober   : trained Prober
        tr_mse   : final training MSE (scalar, in raw units^2)
        ts_mse   : final test     MSE (scalar, in raw units^2)
    """
    loader = torch.utils.data.DataLoader(
        ProbeDataset(z_train, y_train), batch_size=batch_size, shuffle=True
    )
    z_test = z_test.to(device)
    y_test = y_test.to(device)
    opt = torch.optim.Adam(prober.parameters(), lr=lr)

    tr_mse = float("nan")
    for _ in range(epochs):
        prober.train()
        run_loss = 0.0
        n = 0
        for z, y in loader:
            z, y = z.to(device), y.to(device)
            opt.zero_grad()
            loss = torch.mean(torch.sum((y - prober(z)).pow(2), dim=1))
            if torch.isnan(loss):
                break
            loss.backward()
            opt.step()
            run_loss += loss.item()
            n += 1
        if n > 0:
            tr_mse = run_loss / n

    prober.eval()
    with torch.no_grad():
        ts_mse = torch.mean(
            torch.sum((y_test - prober(z_test)).pow(2), dim=1)
        ).item()

    return prober, tr_mse, ts_mse


def predict_prober(prober, z, batch_size=4096, device="cuda"):
    """
    Run a trained prober on latent codes in mini-batches.

    Args:
        prober     : trained Prober
        z          : CPU tensor of latent codes, shape (N, dz)
        batch_size : evaluation batch size
        device     : torch device string

    Returns:
        pred : CPU tensor of predictions, shape (N, Dy)
    """
    prober.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            zb = z[start : start + batch_size].to(device)
            parts.append(prober(zb).cpu())
    return torch.cat(parts, dim=0)


def wrapped_theta_rmse_deg_from_cos_sin(pred_cos_sin, true_theta):
    """
    Convert predicted [cos(theta), sin(theta)] vectors to angles and compute
    wrapped angular RMSE in degrees.
    """
    pred_cos_sin = torch.as_tensor(pred_cos_sin).float()
    true_theta = torch.as_tensor(true_theta).float().reshape(-1)

    pred_theta = torch.atan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])
    dtheta = torch.atan2(
        torch.sin(pred_theta - true_theta),
        torch.cos(pred_theta - true_theta),
    )
    return float(torch.rad2deg(torch.sqrt(torch.mean(dtheta.pow(2)))).item())


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_dataset(model, data, num_frames, device="cuda", batch_size=4096):
    """
    Encode all (trial, window) pairs from frame data through the DYSIB encoder.
    Returns the posterior MEAN (not a sample) for downstream probing / rollout.

    Args:
        model      : DYSIB instance
        data       : (n_trials, T, D) float tensor
        num_frames : window size (numF)
        device     : torch device string
        batch_size : encoding mini-batch size

    Returns:
        zx   : (n_trials * numD, dz) CPU tensor  — mean latent codes, ordered
               by (trial, window) with trial as the slow index
        numD : number of windows per trial = T - 2*num_frames + 1
    """
    numD    = data.shape[1] - 2 * num_frames + 1
    dataset = FrameDataset(data, num_frames)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    zx_parts = []
    with torch.no_grad():
        for x, _ in loader:
            mean_z, _, _ = model.encoder(x.to(device))
            zx_parts.append(mean_z.cpu())

    return torch.cat(zx_parts, dim=0), numD


# ─────────────────────────────────────────────────────────────────────────────
# Latent rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_latent(model, zx, num_frames, numD, device="cuda",
                   num_repeats=40, extra_steps=0):
    """
    Roll out the latent dynamics decoder forward in time from the first window
    of each test trajectory, comparing against the ground-truth latent codes.

    Greedy rollout  : z_{t+1} = z_t + mean_delta(z_t)
    Stochastic      : z_{t+1} = z_t + mean_delta + exp(0.5*logvar) * eps

    Args:
        model       : DYSIB (encoder + delta_decoder)
        zx          : (n_trials * numD, dz) CPU tensor — test-set latent codes
        num_frames  : numF
        numD        : windows per trial
        device      : torch device string
        num_repeats : stochastic samples per trajectory (for mean / std)
        extra_steps : prediction steps beyond the trajectory length

    Returns dict with:
        what_frames  : (n_steps+1,) int array — frame index of each rollout step
        greedy_path  : (n_steps+1, n_trials, dz) float array
        pred_mean    : (n_steps+1, n_trials, dz) float array — mean over stochastic samples
        pred_std     : (n_steps+1, n_trials, dz) float array — std  over stochastic samples
        delta_greedy : (n_compare,) float array — mean RMS in latent space (greedy vs truth)
        delta_pred   : (n_compare,) float array — mean RMS in latent space (stochastic vs truth)
    """
    model.eval()
    n_trials = zx.shape[0] // numD
    dz       = zx.shape[1]
    n_steps  = numD // num_frames + extra_steps

    what_frames = np.arange(n_steps + 1) * num_frames

    # Ground truth: every num_frames-th window per trial (aligned to rollout steps)
    truth = zx.reshape(n_trials, numD, dz)[:, ::num_frames, :].numpy()
    # truth : (n_trials, n_truth_steps, dz)   where n_truth_steps = numD // num_frames + 1

    # Starting latent code: first window of each test trajectory
    start = zx[::numD, :].to(device)   # (n_trials, dz)

    greedy_steps = [start.cpu().numpy()]

    # Stochastic: num_repeats copies per trial
    s = start.unsqueeze(1).expand(-1, num_repeats, -1).reshape(-1, dz).to(device)
    stoch_steps  = [s.cpu().numpy().reshape(n_trials, num_repeats, dz)]

    g = start.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            # greedy step
            delta_g, _ = model.delta_decoder(g)
            g = g + delta_g
            greedy_steps.append(g.cpu().numpy())

            # stochastic step
            delta_s, logvar_s = model.delta_decoder(s)
            noise = torch.randn_like(s)
            s = s + delta_s + (0.5 * logvar_s).exp() * noise
            stoch_steps.append(s.cpu().numpy().reshape(n_trials, num_repeats, dz))

    greedy = np.stack(greedy_steps, axis=0)    # (n_steps+1, n_trials, dz)
    stoch  = np.stack(stoch_steps,  axis=0)    # (n_steps+1, n_trials, num_repeats, dz)

    pred_mean = stoch.mean(axis=2)   # (n_steps+1, n_trials, dz)
    pred_std  = stoch.std(axis=2)    # (n_steps+1, n_trials, dz)

    # Compare against ground truth in latent space
    n_compare = min(n_steps + 1, truth.shape[1])
    truth_c   = truth[:, :n_compare, :]                     # (n_trials, n_compare, dz)
    greedy_c  = greedy[:n_compare].transpose(1, 0, 2)       # (n_trials, n_compare, dz)
    pred_c    = pred_mean[:n_compare].transpose(1, 0, 2)    # (n_trials, n_compare, dz)

    delta_greedy = np.mean(
        np.sqrt(np.sum((truth_c - greedy_c) ** 2, axis=2)), axis=0
    )  # (n_compare,)
    delta_pred = np.mean(
        np.sqrt(np.sum((truth_c - pred_c) ** 2, axis=2)), axis=0
    )  # (n_compare,)

    return {
        "what_frames":  what_frames[:n_compare],
        "greedy_path":  greedy,
        "pred_mean":    pred_mean,
        "pred_std":     pred_std,
        "delta_greedy": delta_greedy,
        "delta_pred":   delta_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Physical-variable RMS along a rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_rms(prober, rollout, target_flat, numD, num_frames, device="cuda"):
    """
    Decode a physical variable along a rollout and compute per-step RMS error.

    Args:
        prober      : trained Prober (on device)
        rollout     : dict returned by rollout_latent()
        target_flat : (n_trials * numD,) CPU tensor of the physical variable,
                      ordered by (trial, window) — same ordering as latent codes
        numD        : windows per trial
        num_frames  : numF (stride for aligning time indices)
        device      : torch device string

    Returns:
        rms_greedy : (n_compare,) ndarray — RMSE per rollout step, greedy path
        rms_pred   : (n_compare,) ndarray — RMSE per rollout step, stochastic mean path
    """
    prober.eval()
    greedy_path = rollout["greedy_path"]   # (n_steps+1, n_trials, dz)
    pred_mean   = rollout["pred_mean"]     # (n_steps+1, n_trials, dz)
    n_steps1, n_trials, dz = greedy_path.shape

    # Align ground truth to rollout time steps (every num_frames-th window)
    truth_2d  = target_flat.reshape(n_trials, numD)[:, ::num_frames].numpy()
    # truth_2d : (n_trials, n_truth_steps)

    n_compare = min(n_steps1, truth_2d.shape[1])

    def _decode(path):
        # path : (n_steps+1, n_trials, dz)
        z = torch.from_numpy(
            path[:n_compare].reshape(-1, dz)   # (n_compare * n_trials, dz)
        ).float().to(device)
        with torch.no_grad():
            vals = prober(z).cpu().numpy()     # (n_compare * n_trials, 1)
        return vals.reshape(n_compare, n_trials)  # (n_compare, n_trials)

    gv = _decode(greedy_path)
    pv = _decode(pred_mean)

    truth_c = truth_2d[:, :n_compare].T    # (n_compare, n_trials)

    rms_greedy = np.sqrt(np.mean((truth_c - gv) ** 2, axis=1))
    rms_pred   = np.sqrt(np.mean((truth_c - pv) ** 2, axis=1))

    return rms_greedy, rms_pred


def rollout_theta_rmse_wrapped(
    prober, rollout, target_flat, numD, num_frames, batch_size=4096, device="cuda"
):
    """
    Decode theta along a rollout using a prober trained on [cos(theta), sin(theta)]
    targets, then compute wrapped angular RMSE in degrees per rollout step.

    Returns:
        rms_greedy : (n_compare,) ndarray — wrapped angular RMSE per step, greedy path
        rms_pred   : (n_compare,) ndarray — wrapped angular RMSE per step, stochastic mean path
    """
    prober.eval()
    greedy_path = rollout["greedy_path"]   # (n_steps+1, n_trials, dz)
    pred_mean   = rollout["pred_mean"]     # (n_steps+1, n_trials, dz)
    n_steps1, n_trials, dz = greedy_path.shape

    truth_theta = torch.as_tensor(target_flat).float().reshape(n_trials, numD)[:, ::num_frames]
    n_compare = min(n_steps1, truth_theta.shape[1])
    truth_theta = truth_theta[:, :n_compare].T  # (n_compare, n_trials)

    def _decode_theta(path):
        z = torch.from_numpy(path[:n_compare].reshape(-1, dz)).float()
        pred_xy = predict_prober(prober, z, batch_size=batch_size, device=device)
        pred_theta = torch.atan2(pred_xy[:, 1], pred_xy[:, 0])
        return pred_theta.reshape(n_compare, n_trials)

    theta_g = _decode_theta(greedy_path)
    theta_p = _decode_theta(pred_mean)

    dtheta_g = torch.atan2(
        torch.sin(theta_g - truth_theta),
        torch.cos(theta_g - truth_theta),
    )
    dtheta_p = torch.atan2(
        torch.sin(theta_p - truth_theta),
        torch.cos(theta_p - truth_theta),
    )

    rms_greedy = torch.rad2deg(torch.sqrt(torch.mean(dtheta_g.pow(2), dim=1))).cpu().numpy()
    rms_pred = torch.rad2deg(torch.sqrt(torch.mean(dtheta_p.pow(2), dim=1))).cpu().numpy()

    return rms_greedy, rms_pred


# ─────────────────────────────────────────────────────────────────────────────
# CLI — train a single DYSIB model and save weights
# ─────────────────────────────────────────────────────────────────────────────

def _run_training(
    *,
    data_path,
    out_path,
    samples_n=1000,
    kz=2,
    num_frames=2,
    rep=0,
    dzf=32,
    hidden=256,
    decoder_hdim=64,
    gamma=0.01,
    alpha=1.0,
    epochs=300,
    lr=1e-4,
    batch_size=1024,
    device="cuda",
):
    """Train a DYSIB model from a preprocessed .npz and save weights to out_path.

    The .npz is expected to have key 'frames' with shape (n_trials, T, D).
    The first `samples_n` trials are used for training; trials [1000:1200] for test.
    """
    from pathlib import Path

    from models import DYSIB

    torch.manual_seed(rep)
    np.random.seed(rep)

    npz = np.load(data_path)
    frames = torch.from_numpy(npz["frames"]).float()
    n_trials, T, D = frames.shape
    print(f"Loaded: {data_path}  shape=({n_trials}, {T}, {D})")

    train_data = frames[:samples_n]
    test_data  = frames[1000:1200]

    train_ds = FrameDataset(train_data, num_frames)
    loader   = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # test batch: concatenate all (past, future) windows from test trials
    test_ds = FrameDataset(test_data, num_frames)
    x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(device)
    y_test = torch.stack([test_ds[i][1] for i in range(len(test_ds))]).to(device)

    model = DYSIB(input_dim=D, dz=kz, dzf=dzf, num_frames=num_frames,
                  gamma=gamma, alpha=alpha, hidden=hidden,
                  decoder_hdim=decoder_hdim).to(device)

    model, hist = train_dysib(model, loader, x_test, y_test,
                              epochs=epochs, lr=lr, device=device)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "params": dict(kz=kz, num_frames=num_frames, samples_n=samples_n, rep=rep,
                       dzf=dzf, hidden=hidden, decoder_hdim=decoder_hdim,
                       gamma=gamma, alpha=alpha, input_dim=D),
        "history": hist,
    }, out_path)
    print(f"Saved: {out_path}")
    return model, hist


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train a single DYSIB model.")
    p.add_argument("--data", type=str, default="data/pendulum_preprocessed.npz",
                   help="Path to preprocessed .npz with key 'frames' of shape (n, T, D)")
    p.add_argument("--out", type=str, required=True, help="Output checkpoint path (.pt)")
    p.add_argument("--samples_n", type=int, default=1000)
    p.add_argument("--kz", type=int, default=2)
    p.add_argument("--num_frames", type=int, default=2)
    p.add_argument("--rep", type=int, default=0, help="Random seed / repetition index")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    _run_training(
        data_path=args.data, out_path=args.out,
        samples_n=args.samples_n, kz=args.kz, num_frames=args.num_frames,
        rep=args.rep, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        gamma=args.gamma, device=args.device,
    )

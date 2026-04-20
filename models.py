"""
Neural network architectures for DYSIB.

DYSIB = Dynamical Symmetric Information Bottleneck.

Classes:
    Encoder        -- per-frame MLP VAE encoder
    DeltaDecoder   -- predicts (mean_delta, logvar) for latent dynamics
    DYSIB          -- full model: shared encoder + delta decoder + NCE MI estimator
    Prober         -- small MLP to decode a physical variable from z (trained separately)
    FrameDataset   -- sliding (past_window, future_window) pairs from trajectory data
    ProbeDataset   -- (z, y) regression dataset for prober training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Per-frame MLP encoder.

    Each of the num_frames input frames is passed through the same 3-layer MLP
    independently (Linear layers broadcast over the frame dimension).  The
    per-frame embeddings are concatenated, then two linear heads produce the
    mean and log-variance of the VAE latent code z ~ N(mean, exp(logvar)).

    Args:
        dz         : latent dimensionality
        dzf        : per-frame embedding size (before concatenation)
        input_dim  : pixel dimension per frame (e.g. 784 for 28x28)
        num_frames : number of consecutive frames in each input window
        hidden     : hidden layer width shared across both hidden layers
    """
    def __init__(self, dz, dzf, input_dim, num_frames, hidden=256):
        super().__init__()
        self.dz = dz
        self.dzf = dzf
        self.num_frames = num_frames
        self.flat_dim = dzf * num_frames   # size after concatenating frame embeddings

        self.dense1 = nn.Linear(input_dim, hidden)
        self.dense2 = nn.Linear(hidden, hidden)
        self.dense3 = nn.Linear(hidden, dzf)   # per-frame output

        self.mean_head   = nn.Linear(self.flat_dim, dz)
        self.logvar_head = nn.Linear(self.flat_dim, dz)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x : (batch, num_frames, D)
        returns (mean_z, logvar_z, z_sample) each of shape (batch, dz)
        """
        h = F.relu(self.dense1(x))                           # (batch, num_frames, hidden)
        h = F.relu(self.dense2(h))                           # (batch, num_frames, hidden)
        h = F.relu(self.dense3(h)).reshape(x.shape[0], -1)  # (batch, num_frames * dzf)
        mean_z   = self.mean_head(h)
        logvar_z = self.logvar_head(h)
        eps = torch.randn_like(mean_z)
        z = mean_z + (0.5 * logvar_z).exp() * eps
        return mean_z, logvar_z, z


class DeltaDecoder(nn.Module):
    """
    Latent dynamics decoder (called δ-predictor in the paper).

    Predicts (mean_delta, logvar) from a past latent code zx, where:
        z_future ≈ zx + mean_delta          (greedy / deterministic rollout)
        z_future ~ N(zx + mean_delta, exp(logvar))  (stochastic rollout)

    Args:
        dz   : latent dimensionality
        hdim : hidden layer width (3 hidden layers)
    """
    def __init__(self, dz, hdim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dz, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, hdim), nn.ReLU(),
        )
        self.mean_head   = nn.Linear(hdim, dz)
        self.logvar_head = nn.Linear(hdim, dz)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        """
        z : (batch, dz)
        returns (mean_delta, logvar) each of shape (batch, dz)
        """
        h = self.mlp(z)
        return self.mean_head(h), self.logvar_head(h)


class DYSIB(nn.Module):
    """
    Dynamical Symmetric Information Bottleneck for frame-sequence data.

    Architecture
    ------------
    * One shared Encoder maps frame windows to latent codes:
          zx ~ q(z | past_frames),   zy ~ q(z | future_frames)
    * A DeltaDecoder maps zx → (mean_delta, logvar), predicting the future code:
          z_future_pred = zx + mean_delta,  variance = exp(logvar)

    Objective (maximise)
    --------------------
        -loss = alpha * I_NCE(zx; zy) - gamma * KL( q(z|X) || N(0,I) )

    The NCE mutual-information lower bound uses a Gaussian critic:
        scores[i,j] = log N(zy_j ; pred_mean_i, exp(pred_logvar_i))
        I_NCE = mean_i [ scores[i,i] - logsumexp_j scores[i,j] + log(batch) ]

    Args:
        input_dim    : per-frame pixel dimension
        dz           : latent dimension
        dzf          : per-frame embedding size inside encoder
        num_frames   : frames per window
        gamma        : weight on the KL compression term (recommend 0.01)
        alpha        : weight on the MI maximisation term (typically 1.0)
        hidden       : encoder MLP hidden width
        decoder_hdim : DeltaDecoder hidden width
    """
    def __init__(self, input_dim, dz=2, dzf=32, num_frames=2,
                 gamma=0.01, alpha=1.0, hidden=256, decoder_hdim=64):
        super().__init__()
        self.encoder       = Encoder(dz=dz, dzf=dzf, input_dim=input_dim,
                                     num_frames=num_frames, hidden=hidden)
        self.delta_decoder = DeltaDecoder(dz=dz, hdim=decoder_hdim)
        self.gamma = gamma
        self.alpha = alpha

    def _nce_info(self, zy, pred_mean, pred_logvar):
        """
        Gaussian InfoNCE lower bound on I(zx; zy).

        scores[i, j] = log N(zy_j ; pred_mean_i, exp(pred_logvar_i))
        I >= mean_i [ scores[i,i] - logsumexp_j(scores[i,j]) + log(batch_size) ]
        """
        batch = zy.shape[0]
        # broadcast to (batch, batch, dz): axis-0 indexes the predictor (i), axis-1 the target (j)
        zy_rep  = zy.unsqueeze(0).expand(batch, -1, -1)          # (B, B, dz): zy_rep[i,j] = zy[j]
        mu_rep  = pred_mean.unsqueeze(1).expand(-1, batch, -1)   # (B, B, dz): mu_rep[i,j]  = mu[i]
        lv_rep  = pred_logvar.unsqueeze(1).expand(-1, batch, -1) # (B, B, dz): lv_rep[i,j]  = lv[i]

        LOG2PI = 1.8378770664093453
        scores = -0.5 * ((zy_rep - mu_rep).pow(2) / lv_rep.exp() + lv_rep + LOG2PI)
        scores = scores.sum(dim=2)  # (B, B)

        return torch.mean(
            scores.diagonal() - scores.logsumexp(dim=1) + torch.tensor(float(batch)).log()
        )

    def forward(self, x, y):
        """
        x, y : (batch, num_frames, D)  — past and future frame windows
        returns (loss, kl_loss, mi_loss, mi_estimate)
        """
        mean_zx, logvar_zx, zx = self.encoder(x)
        kl_x = torch.mean(
            torch.sum(-0.5 * (1 + logvar_zx - mean_zx.pow(2) - logvar_zx.exp()), dim=1)
        )
        mean_zy, logvar_zy, zy = self.encoder(y)
        kl_y = torch.mean(
            torch.sum(-0.5 * (1 + logvar_zy - mean_zy.pow(2) - logvar_zy.exp()), dim=1)
        )

        delta, delta_logvar = self.delta_decoder(zx)
        pred_mean = zx + delta   # residual prediction: future ≈ past + delta

        mi      = self._nce_info(zy, pred_mean, delta_logvar)
        kl      = self.gamma * (kl_x + kl_y)
        mi_term = self.alpha * mi
        loss    = kl - mi_term

        return loss, kl, mi_term, mi


class Prober(nn.Module):
    """
    Small MLP that decodes a scalar physical variable from a latent code.
    Trained separately (DYSIB weights frozen) with MSE loss.

    Args:
        dz         : latent dimension (input size)
        output_dim : 1 for scalar variables (theta, omega, energy)
        hdim       : hidden layer width (3 hidden layers)
    """
    def __init__(self, dz, output_dim=1, hdim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz, hdim),    nn.ReLU(),
            nn.Linear(hdim, hdim),  nn.ReLU(),
            nn.Linear(hdim, hdim),  nn.ReLU(),
            nn.Linear(hdim, output_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class FrameDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset over trajectory data.

    Each item is a (past_window, future_window) pair of shape (num_frames, D).
    For a trajectory of length T, there are numD = T - 2*num_frames + 1 valid
    start positions (windows 0..numD-1 as past; numD..2*numD-1 as future).

    data       : (n_trials, T, D) float tensor
    num_frames : window size
    """
    def __init__(self, data, num_frames):
        self.data       = data
        self.num_frames = num_frames
        self.n_trials   = data.shape[0]
        self.T          = data.shape[1]
        self.numD       = self.T - 2 * num_frames + 1
        assert self.numD > 0, f"Not enough frames: T={self.T}, num_frames={num_frames}"

    def __len__(self):
        return self.n_trials * self.numD

    def __getitem__(self, idx):
        trial = idx // self.numD
        t     = idx  % self.numD
        past   = self.data[trial, t           : t + self.num_frames,           :]
        future = self.data[trial, t + self.num_frames : t + 2 * self.num_frames, :]
        return past, future


class ProbeDataset(torch.utils.data.Dataset):
    """Simple (z, y) regression dataset for prober training."""
    def __init__(self, z, y):
        self.z = z
        self.y = y

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]

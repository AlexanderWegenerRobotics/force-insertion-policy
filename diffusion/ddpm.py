import torch
import torch.nn as nn
import numpy as np


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class BranchMLP(nn.Module):
    """
    Per-input encoder branch.
    Maps one input stream to a shared feature size.
    """
    def __init__(self, in_dim: int, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBlock(nn.Module):
    """
    Residual MLP block that preserves feature dimension.
    """
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class NoiseEstimator(nn.Module):
    def __init__(self, obs_dim: int = 18, action_dim: int = 6, branch_feature_dim: int = 128, hidden_dim: int = 512, timestep_embed_dim: int = 128):
        super().__init__()

        self.timestep_embed = SinusoidalEmbedding(timestep_embed_dim)

        # Per-input branches
        self.obs_prev_encoder = BranchMLP(obs_dim, branch_feature_dim)
        self.obs_curr_encoder = BranchMLP(obs_dim, branch_feature_dim)
        self.noisy_action_encoder = BranchMLP(action_dim, branch_feature_dim)
        self.time_encoder = BranchMLP(timestep_embed_dim, branch_feature_dim)

        # After concatenating 4 branches
        fused_dim = 4 * branch_feature_dim

        # Residual backbone
        self.residual_block_1 = MLPBlock(fused_dim, hidden_dim)
        self.residual_block_2 = MLPBlock(fused_dim, hidden_dim)
        self.residual_block_3 = MLPBlock(fused_dim, hidden_dim)

        # Final prediction head
        self.output_head = nn.Linear(fused_dim, action_dim)

    def forward(self, obs_prev: torch.Tensor, obs_curr: torch.Tensor, action_noisy: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        tau_emb = self.timestep_embed(tau)

        z_obs_prev = self.obs_prev_encoder(obs_prev)
        z_obs_curr = self.obs_curr_encoder(obs_curr)
        z_action = self.noisy_action_encoder(action_noisy)
        z_time = self.time_encoder(tau_emb)

        z_fused = torch.cat([z_obs_prev, z_obs_curr, z_action, z_time], dim=-1)

        h = self.residual_block_1(z_fused)
        h = self.residual_block_2(h)
        h = self.residual_block_3(h)

        eps_hat = self.output_head(h)
        return eps_hat


class DDPMSchedule:
    def __init__(self, T=50, beta_start=1e-4, beta_end=1e-2, device="cpu"):
        """Precompute variance schedule: alpha_t = 1 - beta_t, alpha^bar_t = product_i^T alpha_i, sigma_t = sqrt(beta_t)"""
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)

    def q_sample(self, a0, tau, noise=None):
        """Forward diffusion: a_t = sqrt(alpha^bar_t) * a_0 + sqrt(1-alpha^bar_t) * eps  (Eq. 2)"""
        if noise is None:
            noise = torch.randn_like(a0)
        ab = self.alpha_bar[tau]
        while ab.dim() < a0.dim():
            ab = ab.unsqueeze(-1)
        return torch.sqrt(ab) * a0 + torch.sqrt(1 - ab) * noise, noise

    def p_sample(self, model, o_prev, o_curr, a_tau, tau):
        """Reverse denoising: a_{t-1} = 1/sqrt(alpha_t) * [a_t - (1-alpha_t)/sqrt(1-alpha^bar_t) * eps] + sigma_t * eps  (Eq. 6)"""
        eps_hat = model(o_prev, o_curr, a_tau, tau)
        alpha = self.alphas[tau]
        alpha_bar = self.alpha_bar[tau]
        sigma = self.sigmas[tau]
        while alpha.dim() < a_tau.dim():
            alpha = alpha.unsqueeze(-1)
            alpha_bar = alpha_bar.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        mean = (1.0 / torch.sqrt(alpha)) * (a_tau - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_hat)
        if tau[0].item() > 0:
            return mean + sigma * torch.randn_like(a_tau)
        return mean

    def to(self, device):
        """Move all schedule tensors to target device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sigmas = self.sigmas.to(device)
        return self


class DDIMSampler:
    """
    Deterministic sampler using a DDPM-trained noise estimator.
    Implements the non-Markovian reverse process from Song et al. (2021)
    with eta=0 (fully deterministic) by default.
    """
    def __init__(self, schedule: DDPMSchedule, num_steps: int = 10, eta: float = 0.0):
        self.schedule = schedule
        self.eta = eta
        self.timesteps = self._make_subsequence(schedule.T, num_steps)

    @staticmethod
    def _make_subsequence(T: int, num_steps: int) -> list:
        """Uniformly spaced subsequence of [0, T-1] in descending order."""
        step_size = T // num_steps
        return list(range(T - 1, -1, -step_size))[:num_steps]

    def set_num_steps(self, num_steps: int):
        """Recompute the timestep subsequence for a new step count."""
        self.timesteps = self._make_subsequence(self.schedule.T, num_steps)

    def p_sample_step(self, model, o_prev, o_curr, a_tau, tau_idx):
        """Single DDIM denoising step from timesteps[tau_idx] to timesteps[tau_idx+1] (or to 0)."""
        t_curr = self.timesteps[tau_idx]
        batch_size = a_tau.shape[0]
        device = a_tau.device

        tau = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)
        eps_hat = model(o_prev, o_curr, a_tau, tau)

        ab_curr = self.schedule.alpha_bar[t_curr]

        # Predict clean action: a0_hat = (a_t - sqrt(1-ab_t) * eps) / sqrt(ab_t)
        a0_hat = (a_tau - torch.sqrt(1 - ab_curr) * eps_hat) / torch.sqrt(ab_curr)

        if tau_idx + 1 < len(self.timesteps):
            t_next = self.timesteps[tau_idx + 1]
            ab_next = self.schedule.alpha_bar[t_next]
        else:
            return a0_hat

        # DDIM stochastic coefficient (eta=0 -> deterministic)
        sigma = self.eta * torch.sqrt((1 - ab_next) / (1 - ab_curr) * (1 - ab_curr / ab_next))

        # Direction pointing toward a_t
        dir_at = torch.sqrt(1 - ab_next - sigma ** 2) * eps_hat

        a_next = torch.sqrt(ab_next) * a0_hat + dir_at
        if self.eta > 0:
            a_next = a_next + sigma * torch.randn_like(a_tau)

        return a_next

    @torch.no_grad()
    def sample(self, model, o_prev, o_curr, shape, device):
        """Full DDIM reverse process: noise -> action in len(self.timesteps) model evaluations."""
        a = torch.randn(shape, device=device)
        for i in range(len(self.timesteps)):
            a = self.p_sample_step(model, o_prev, o_curr, a, i)
        return a
import torch
import torch.nn as nn
import numpy as np


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class NoiseEstimator(nn.Module):
    def __init__(self, obs_dim=18, action_dim=6, hidden_dim=512, timestep_embed_dim=128):
        super().__init__()
        self.timestep_embed = SinusoidalEmbedding(timestep_embed_dim)
        input_dim = obs_dim * 2 + action_dim + timestep_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.residual_proj = nn.Linear(input_dim, action_dim) if input_dim != action_dim else nn.Identity()

    def forward(self, o_prev, o_curr, a_noised, tau):
        tau_emb = self.timestep_embed(tau)
        x = torch.cat([o_prev, o_curr, a_noised, tau_emb], dim=-1)
        return self.net(x) + self.residual_proj(x)


class DDPMSchedule:
    def __init__(self, T=50, beta_start=1e-4, beta_end=1e-2, device="cpu"):
        """Precompute variance schedule: alpha_t = 1 - beta_t, alpha^bar_t = product_i^T alpha_i, alpha_t = sqrt(beta_t)"""
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)

    def q_sample(self, a0, tau, noise=None):
        """Forward diffusion: alpha_t = sqrt(alpha^bar_t) * a_0 + sqrt(1-alpha^bar_t) * eps  (Eq. 2)"""
        if noise is None:
            noise = torch.randn_like(a0)
        ab = self.alpha_bar[tau]
        while ab.dim() < a0.dim():
            ab = ab.unsqueeze(-1)
        return torch.sqrt(ab) * a0 + torch.sqrt(1 - ab) * noise, noise

    def p_sample(self, model, o_prev, o_curr, a_tau, tau):
        """Reverse denoising: a_{t-1} = 1/sqrt(alpha_t) * [a_t - (1-alpha_t)/sqrt(1-alpha^bar_t) * eps] + alpha_t * eps  (Eq. 6)"""
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
import torch
import torch.nn as nn
import numpy as np


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
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
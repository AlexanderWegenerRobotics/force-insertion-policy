import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class BranchMLP(nn.Module):
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


class ConfigurableNoiseEstimator(nn.Module):
    def __init__(
        self,
        obs_prev_dim: int = 18,
        obs_curr_dim: int = 18,
        action_dim: int = 6,
        branch_feature_dim: int = 128,
        hidden_dim: int = 512,
        timestep_embed_dim: int = 128,
        num_residual_blocks: int = 3,
        share_obs_encoder: bool = False,
        fuse_observations_early: bool = False,
    ):
        super().__init__()
        self.fuse_observations_early = fuse_observations_early
        self.share_obs_encoder = share_obs_encoder

        self.timestep_embed = SinusoidalEmbedding(timestep_embed_dim)
        self.noisy_action_encoder = BranchMLP(action_dim, branch_feature_dim)
        self.time_encoder = BranchMLP(timestep_embed_dim, branch_feature_dim)

        if fuse_observations_early:
            self.obs_encoder = BranchMLP(obs_prev_dim + obs_curr_dim, branch_feature_dim)
            num_fused_branches = 3
        elif share_obs_encoder:
            if obs_prev_dim != obs_curr_dim:
                raise ValueError("Shared observation encoder requires equal observation widths.")
            self.obs_encoder = BranchMLP(obs_prev_dim, branch_feature_dim)
            num_fused_branches = 4
        else:
            self.obs_prev_encoder = BranchMLP(obs_prev_dim, branch_feature_dim)
            self.obs_curr_encoder = BranchMLP(obs_curr_dim, branch_feature_dim)
            num_fused_branches = 4

        fused_dim = num_fused_branches * branch_feature_dim
        self.residual_blocks = nn.ModuleList(
            [MLPBlock(fused_dim, hidden_dim) for _ in range(num_residual_blocks)]
        )
        self.output_head = nn.Linear(fused_dim + 2 * branch_feature_dim, action_dim)

    def _encode_observations(self, obs_prev: torch.Tensor, obs_curr: torch.Tensor):
        if self.fuse_observations_early:
            return [self.obs_encoder(torch.cat([obs_prev, obs_curr], dim=-1))]
        if self.share_obs_encoder:
            return [self.obs_encoder(obs_prev), self.obs_encoder(obs_curr)]
        return [self.obs_prev_encoder(obs_prev), self.obs_curr_encoder(obs_curr)]

    def forward(self, obs_prev, obs_curr, action_noisy, tau):
        tau_emb = self.timestep_embed(tau)
        obs_features = self._encode_observations(obs_prev, obs_curr)
        z_action = self.noisy_action_encoder(action_noisy)
        z_time = self.time_encoder(tau_emb)
        h = torch.cat([*obs_features, z_action, z_time], dim=-1)
        for block in self.residual_blocks:
            h = block(h)
        # Re-inject action and time encodings so conditioning is never lost
        h = torch.cat([h, z_action, z_time], dim=-1)
        return self.output_head(h)


class NoiseEstimator(ConfigurableNoiseEstimator):
    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 6,
        branch_feature_dim: int = 128,
        hidden_dim: int = 512,
        timestep_embed_dim: int = 128,
    ):
        super().__init__(
            obs_prev_dim=obs_dim,
            obs_curr_dim=obs_dim,
            action_dim=action_dim,
            branch_feature_dim=branch_feature_dim,
            hidden_dim=hidden_dim,
            timestep_embed_dim=timestep_embed_dim,
            num_residual_blocks=3,
        )


class SimpleMLP(nn.Module):
    """Flat MLP with no branches and no residual blocks — lowest inference cost."""
    def __init__(self, obs_dim: int = 18, action_dim: int = 6,
                 timestep_embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.timestep_embed = SinusoidalEmbedding(timestep_embed_dim)
        in_dim = 2 * obs_dim + action_dim + timestep_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs_prev: torch.Tensor, obs_curr: torch.Tensor,
                action_noisy: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        tau_emb = self.timestep_embed(tau)
        x = torch.cat([obs_prev, obs_curr, action_noisy, tau_emb], dim=-1)
        return self.net(x)


class CompactWidthNoiseEstimator(ConfigurableNoiseEstimator):
    def __init__(self, obs_dim: int = 18, action_dim: int = 6):
        super().__init__(obs_dim, obs_dim, action_dim, 96, 256, 96, 3)


class ShallowBackboneNoiseEstimator(ConfigurableNoiseEstimator):
    def __init__(self, obs_dim: int = 18, action_dim: int = 6):
        super().__init__(obs_dim, obs_dim, action_dim, 128, 384, 128, 2)


class SharedObservationNoiseEstimator(ConfigurableNoiseEstimator):
    def __init__(self, obs_dim: int = 18, action_dim: int = 6):
        super().__init__(obs_dim, obs_dim, action_dim, 128, 320, 128, 2, share_obs_encoder=True)


class FusedObservationNoiseEstimator(ConfigurableNoiseEstimator):
    def __init__(self, obs_dim: int = 18, action_dim: int = 6):
        super().__init__(
            obs_dim,
            obs_dim,
            action_dim,
            128,
            320,
            128,
            2,
            fuse_observations_early=True,
        )


class ExtendedHistoryNoiseEstimator(ConfigurableNoiseEstimator):
    def __init__(self, base_obs_dim: int = 18, action_dim: int = 6, history_length: int = 3):
        prev_history_dim = (history_length - 1) * base_obs_dim
        super().__init__(
            prev_history_dim,
            base_obs_dim,
            action_dim,
            96,
            256,
            96,
            2,
            fuse_observations_early=True,
        )


class FusedObservationLargeNoiseEstimator(ConfigurableNoiseEstimator):
    def __init__(self, obs_dim: int = 18, action_dim: int = 6):
        super().__init__(
            obs_dim, obs_dim, action_dim,
            branch_feature_dim=160,
            hidden_dim=512,
            timestep_embed_dim=160,
            num_residual_blocks=3,
            fuse_observations_early=True,
        )


MODEL_VARIANTS = {
    "baseline": NoiseEstimator,
    "simple_mlp": SimpleMLP,
    "compact_width": CompactWidthNoiseEstimator,
    "shallow_backbone": ShallowBackboneNoiseEstimator,
    "shared_observation": SharedObservationNoiseEstimator,
    "fused_observation": FusedObservationNoiseEstimator,
    "fused_observation_large": FusedObservationLargeNoiseEstimator,
    "extended_history": ExtendedHistoryNoiseEstimator,
}


def build_noise_estimator(variant: str = "baseline", **kwargs) -> nn.Module:
    if variant not in MODEL_VARIANTS:
        valid = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown model variant '{variant}'. Valid options: {valid}")
    return MODEL_VARIANTS[variant](**kwargs)


class DDPMSchedule:
    def __init__(self, T=50, beta_start=1e-4, beta_end=1e-2, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)

    def q_sample(self, a0, tau, noise=None):
        if noise is None:
            noise = torch.randn_like(a0)
        ab = self.alpha_bar[tau]
        while ab.dim() < a0.dim():
            ab = ab.unsqueeze(-1)
        return torch.sqrt(ab) * a0 + torch.sqrt(1 - ab) * noise, noise

    def p_sample(self, model, o_prev, o_curr, a_tau, tau):
        eps_hat = model(o_prev, o_curr, a_tau, tau)
        alpha = self.alphas[tau]
        alpha_bar = self.alpha_bar[tau]
        sigma = self.sigmas[tau]
        while alpha.dim() < a_tau.dim():
            alpha = alpha.unsqueeze(-1)
            alpha_bar = alpha_bar.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        mean = (1.0 / torch.sqrt(alpha)) * (
            a_tau - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_hat
        )
        if tau[0].item() > 0:
            return mean + sigma * torch.randn_like(a_tau)
        return mean

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sigmas = self.sigmas.to(device)
        return self


class DDIMSampler:
    def __init__(self, schedule: DDPMSchedule, num_steps: int = 10, eta: float = 0.0):
        self.schedule = schedule
        self.eta = eta
        self.timesteps = self._make_subsequence(schedule.T, num_steps)

    @staticmethod
    def _make_subsequence(T: int, num_steps: int) -> list:
        step_size = max(T // num_steps, 1)
        timesteps = list(range(T - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)
        return timesteps[:num_steps]

    def set_num_steps(self, num_steps: int):
        self.timesteps = self._make_subsequence(self.schedule.T, num_steps)

    def p_sample_step(self, model, o_prev, o_curr, a_tau, tau_idx):
        t_curr = self.timesteps[tau_idx]
        batch_size = a_tau.shape[0]
        tau = torch.full((batch_size,), t_curr, device=a_tau.device, dtype=torch.long)
        eps_hat = model(o_prev, o_curr, a_tau, tau)
        ab_curr = self.schedule.alpha_bar[t_curr]
        a0_hat = (a_tau - torch.sqrt(1 - ab_curr) * eps_hat) / torch.sqrt(ab_curr)
        if tau_idx + 1 >= len(self.timesteps):
            return a0_hat
        t_next = self.timesteps[tau_idx + 1]
        ab_next = self.schedule.alpha_bar[t_next]
        sigma = self.eta * torch.sqrt((1 - ab_next) / (1 - ab_curr) * (1 - ab_curr / ab_next))
        dir_at = torch.sqrt(1 - ab_next - sigma**2) * eps_hat
        a_next = torch.sqrt(ab_next) * a0_hat + dir_at
        if self.eta > 0:
            a_next = a_next + sigma * torch.randn_like(a_tau)
        return a_next

    @torch.no_grad()
    def sample(self, model, o_prev, o_curr, shape, device):
        a = torch.randn(shape, device=device)
        for i in range(len(self.timesteps)):
            a = self.p_sample_step(model, o_prev, o_curr, a, i)
        return a

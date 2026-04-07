import torch
import torch.nn as nn


class ConditionalVAE(nn.Module):
    def __init__(self, obs_dim=18, action_dim=6, hidden_dim=512, latent_dim=16):
        super().__init__()
        cond_dim = obs_dim * 2
        encoder_input_dim = cond_dim + action_dim
        decoder_input_dim = cond_dim + latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.decoder_residual = (
            nn.Linear(decoder_input_dim, action_dim)
            if decoder_input_dim != action_dim
            else nn.Identity()
        )

    def encode(self, o_prev, o_curr, action):
        x = torch.cat([o_prev, o_curr, action], dim=-1)
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, o_prev, o_curr, z):
        x = torch.cat([o_prev, o_curr, z], dim=-1)
        return self.decoder(x) + self.decoder_residual(x)

    def forward(self, o_prev, o_curr, action):
        mu, logvar = self.encode(o_prev, o_curr, action)
        z = self.reparameterize(mu, logvar)
        action_pred = self.decode(o_prev, o_curr, z)
        return action_pred, mu, logvar

    @staticmethod
    def kl_divergence(mu, logvar):
        return 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=-1))

    def sample(self, o_prev, o_curr, z=None):
        if z is None:
            latent_dim = self.mu_head.out_features
            z = torch.randn(o_prev.shape[0], latent_dim, device=o_prev.device, dtype=o_prev.dtype)
        return self.decode(o_prev, o_curr, z)

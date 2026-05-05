import torch
import torch.nn as nn


def _input_dim(obs_dim: int, action_dim: int, include_prev_action: bool) -> int:
    return obs_dim + (action_dim if include_prev_action else 0)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class GRUBC(nn.Module):
    """GRU behavioral cloning policy for feed-forward wrench prediction.

    Input shape is (batch, seq_len, obs_dim + action_dim) when previous actions
    are enabled, otherwise (batch, seq_len, obs_dim). The model predicts the
    normalized current 6D action.
    """

    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.05,
        include_prev_action: bool = True,
    ):
        super().__init__()
        if action_dim != 6:
            raise ValueError("GRUBC currently expects a 6D wrench action.")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.include_prev_action = include_prev_action
        input_dim = _input_dim(obs_dim, action_dim, include_prev_action)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.trunk = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=dropout),
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fxy_head = nn.Linear(hidden_dim, 2)
        self.fz_head = nn.Linear(hidden_dim, 1)
        self.torque_head = nn.Linear(hidden_dim, 3)

    def forward(self, seq):
        x = self.input_proj(seq)
        _, h_n = self.gru(x)
        h = self.trunk(h_n[-1])
        return torch.cat(
            [
                self.fxy_head(h),
                self.fz_head(h),
                self.torque_head(h),
            ],
            dim=-1,
        )


class TransformerBC(nn.Module):
    """Transformer encoder policy for history-conditioned wrench prediction."""

    def __init__(
        self,
        obs_dim: int = 18,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.05,
        include_prev_action: bool = True,
        num_heads: int = 4,
        ff_dim: int | None = None,
        max_seq_len: int = 128,
        pooling: str = "last",
    ):
        super().__init__()
        if action_dim != 6:
            raise ValueError("TransformerBC currently expects a 6D wrench action.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be either 'last' or 'mean'.")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.include_prev_action = include_prev_action
        self.max_seq_len = max_seq_len
        self.pooling = pooling
        input_dim = _input_dim(obs_dim, action_dim, include_prev_action)
        ff_dim = ff_dim or hidden_dim * 4

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trunk = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fxy_head = nn.Linear(hidden_dim, 2)
        self.fz_head = nn.Linear(hidden_dim, 1)
        self.torque_head = nn.Linear(hidden_dim, 3)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, seq):
        if seq.shape[1] > self.max_seq_len:
            raise ValueError(f"Sequence length {seq.shape[1]} exceeds max_seq_len={self.max_seq_len}.")

        x = self.input_proj(seq)
        x = x + self.pos_embed[:, : seq.shape[1]]
        x = self.encoder(x)
        h = x[:, -1] if self.pooling == "last" else x.mean(dim=1)
        h = self.trunk(h)
        return torch.cat(
            [
                self.fxy_head(h),
                self.fz_head(h),
                self.torque_head(h),
            ],
            dim=-1,
        )


def build_transformer_model(cfg: dict) -> nn.Module:
    model_type = cfg.get("model_type", cfg.get("architecture", "gru"))
    common = {
        "hidden_dim": int(cfg.get("hidden_dim", 256)),
        "num_layers": int(cfg.get("num_layers", 2)),
        "dropout": float(cfg.get("dropout", 0.05)),
        "include_prev_action": bool(cfg.get("include_prev_action", True)),
    }
    if model_type in {"gru", "temporal_bc"}:
        return GRUBC(**common)
    if model_type in {"transformer", "temporal_transformer"}:
        return TransformerBC(
            **common,
            num_heads=int(cfg.get("num_heads", 4)),
            ff_dim=int(cfg.get("ff_dim", common["hidden_dim"] * 4)),
            max_seq_len=int(cfg.get("max_seq_len", max(128, int(cfg.get("seq_len", 20))))),
            pooling=cfg.get("pooling", "last"),
        )
    raise ValueError(f"Unknown transformer model_type: {model_type}")

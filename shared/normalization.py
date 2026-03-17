import numpy as np
import h5py
import yaml
from pathlib import Path


def compute_normalization_stats(data_dir: str, success_only: bool = True) -> dict:
    """Compute per-channel mean/std over all timesteps from training episodes."""
    data_dir = Path(data_dir)
    with open(data_dir / "dataset_index.yaml", "r") as f:
        index = yaml.safe_load(f)

    obs_accum = []
    action_accum = []

    for entry in index:
        if success_only and not entry["success"]:
            continue

        filepath = data_dir / entry["path"]
        with h5py.File(filepath, "r") as ep:
            f_ext = ep["obs/f_ext"][:]
            f_internal = ep["obs/f_internal"][:]
            ee_vel = ep["obs/ee_velocity"][:]
            obs = np.concatenate([f_ext, f_internal, ee_vel], axis=1)  # (T, 18)
            action = ep["action/Fff"][:]  # (T, 6)

            obs_accum.append(obs)
            action_accum.append(action)

    all_obs = np.concatenate(obs_accum, axis=0)
    all_actions = np.concatenate(action_accum, axis=0)

    return {
        "obs_mean": all_obs.mean(axis=0).tolist(),
        "obs_std": all_obs.std(axis=0).tolist(),
        "action_mean": all_actions.mean(axis=0).tolist(),
        "action_std": all_actions.std(axis=0).tolist(),
        "n_episodes": len(obs_accum),
        "n_timesteps": int(all_obs.shape[0]),
    }


def save_normalization_stats(stats: dict, path: str):
    """Write stats dict to YAML."""
    with open(path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)


def load_normalization_stats(path: str) -> dict:
    """Load stats from YAML and convert lists back to numpy arrays."""
    with open(path, "r") as f:
        stats = yaml.safe_load(f)

    return {
        "obs_mean": np.array(stats["obs_mean"], dtype=np.float32),
        "obs_std": np.array(stats["obs_std"], dtype=np.float32),
        "action_mean": np.array(stats["action_mean"], dtype=np.float32),
        "action_std": np.array(stats["action_std"], dtype=np.float32),
    }


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean unit-variance normalization with epsilon for numerical stability."""
    return (x - mean) / (std + eps)


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse of normalize — maps back to raw scale."""
    return x * (std + eps) + mean
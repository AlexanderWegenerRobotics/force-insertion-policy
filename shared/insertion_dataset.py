import numpy as np
import h5py
import yaml
import torch
from pathlib import Path
from torch.utils.data import Dataset


class InsertionDataset(Dataset):
    def __init__(self, cfg, split="train"):
        data_dir = Path(cfg["data_dir"])
        with open(data_dir / "dataset_index.yaml", "r") as f:
            index = yaml.safe_load(f)

        successful = [e for e in index if e["success"]]
        rng = np.random.RandomState(cfg.get("val_seed", 42))
        perm = rng.permutation(len(successful))
        n_val = int(len(successful) * cfg.get("val_ratio", 0.2))

        if split == "train":
            indices = perm[n_val:]
        else:
            indices = perm[:n_val]

        with open(cfg["norm_stats"], "r") as f:
            stats = yaml.safe_load(f)
        self.obs_mean = np.array(stats["obs_mean"], dtype=np.float32)
        self.obs_std = np.array(stats["obs_std"], dtype=np.float32) + 1e-6
        self.act_mean = np.array(stats["action_mean"], dtype=np.float32)
        self.act_std = np.array(stats["action_std"], dtype=np.float32) + 1e-6

        self.observations = []
        self.actions = []
        self.timestep_index = []

        for i, idx in enumerate(indices):
            entry = successful[idx]
            with h5py.File(data_dir / entry["path"], "r") as f:
                obs = np.concatenate([f["obs/f_ext"][:], f["obs/f_internal"][:], f["obs/ee_velocity"][:]], axis=1).astype(np.float32)
                act = f["action/Fff"][:].astype(np.float32)

            obs = (obs - self.obs_mean) / self.obs_std
            act = (act - self.act_mean) / self.act_std

            self.observations.append(obs)
            self.actions.append(act)
            for t in range(len(obs)):
                self.timestep_index.append((i, t))

    def __len__(self):
        return len(self.timestep_index)

    def __getitem__(self, idx):
        ep_idx, t = self.timestep_index[idx]
        obs = self.observations[ep_idx]
        o_curr = obs[t]
        o_prev = obs[t - 1] if t > 0 else o_curr.copy()
        action = self.actions[ep_idx][t]
        return torch.from_numpy(o_prev), torch.from_numpy(o_curr), torch.from_numpy(action)

    def denormalize_action(self, action):
        return action * self.act_std + self.act_mean
    
if __name__ == "__main__":
    import yaml

    with open("configs/data_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    train_ds = InsertionDataset(cfg, split="train")
    val_ds = InsertionDataset(cfg, split="val")

    print(f"Train: {len(train_ds)} timesteps, Val: {len(val_ds)} timesteps")

    o_prev, o_curr, action = train_ds[0]
    print(f"o_prev: {o_prev.shape}, o_curr: {o_curr.shape}, action: {action.shape}")
    print(f"o_prev dtype: {o_prev.dtype}")

    o_prev_0, o_curr_0, _ = train_ds[0]
    print(f"Episode boundary check — o_prev == o_curr at t=0: {torch.allclose(o_prev_0, o_curr_0)}")
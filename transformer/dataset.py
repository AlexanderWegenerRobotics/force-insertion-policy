from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class SequenceInsertionDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.seq_len = int(cfg.get("seq_len", 20))
        self.include_prev_action = bool(cfg.get("include_prev_action", True))

        data_dir = Path(cfg["data_dir"])
        with open(data_dir / "dataset_index.yaml", "r", encoding="utf-8") as f:
            index = yaml.safe_load(f)

        success_only = bool(cfg.get("success_only", True))
        entries = [e for e in index if (e["success"] or not success_only)]
        rng = np.random.RandomState(cfg.get("val_seed", 42))
        perm = rng.permutation(len(entries))
        n_val = int(len(entries) * cfg.get("val_ratio", 0.2))
        selected = perm[n_val:] if split == "train" else perm[:n_val]

        with open(cfg["norm_stats"], "r", encoding="utf-8") as f:
            stats = yaml.safe_load(f)
        self.obs_mean = np.array(stats["obs_mean"], dtype=np.float32)
        self.obs_std = np.array(stats["obs_std"], dtype=np.float32) + 1e-6
        self.act_mean = np.array(stats["action_mean"], dtype=np.float32)
        self.act_std = np.array(stats["action_std"], dtype=np.float32) + 1e-6

        self.episodes = []
        self.index = []
        for idx in selected:
            entry = entries[idx]
            with h5py.File(data_dir / entry["path"], "r") as f:
                obs = np.concatenate(
                    [
                        f["obs/f_ext"][:],
                        f["obs/f_internal"][:],
                        f["obs/ee_velocity"][:],
                    ],
                    axis=1,
                ).astype(np.float32)
                act = f["action/Fff"][:].astype(np.float32)

            obs = (obs - self.obs_mean) / self.obs_std
            act = (act - self.act_mean) / self.act_std

            if self.include_prev_action:
                prev_act = np.zeros_like(act)
                prev_act[1:] = act[:-1]
                features = np.concatenate([obs, prev_act], axis=1).astype(np.float32)
            else:
                features = obs.astype(np.float32)

            if self.seq_len > 1:
                pad = np.repeat(features[:1], self.seq_len - 1, axis=0)
                if self.include_prev_action:
                    pad[:, obs.shape[1]:] = 0.0
                features = np.concatenate([pad, features], axis=0)

            episode_i = len(self.episodes)
            self.episodes.append((features, act))
            self.index.extend((episode_i, t) for t in range(len(obs)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        episode_i, t = self.index[idx]
        features, act = self.episodes[episode_i]
        seq = features[t:t + self.seq_len]
        return torch.from_numpy(seq), torch.from_numpy(act[t])

    def materialize_windows(self, device=None):
        seqs = []
        acts = []
        for features, act in self.episodes:
            features_t = torch.from_numpy(features)
            windows = features_t.unfold(0, self.seq_len, 1).transpose(1, 2).contiguous()
            seqs.append(windows)
            acts.append(torch.from_numpy(act))

        seq = torch.cat(seqs, dim=0)
        action = torch.cat(acts, dim=0)
        if device is not None:
            seq = seq.to(device)
            action = action.to(device)
        return seq, action

    def denormalize_action(self, action):
        return action * self.act_std + self.act_mean

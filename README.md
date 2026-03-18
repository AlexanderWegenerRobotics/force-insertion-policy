# force-insertion-policy

Policy training for tight-clearance peg-in-hole insertion using demonstration data collected with [force-insertion-sim](https://github.com/AlexanderWegenerRobotics/force-insertion-sim).

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AlexanderWegenerRobotics/force-insertion-policy.git
cd force-insertion-policy
pip install -e .
```

### 2. Get the data

Data is not tracked in git. Get the dataset from Alex and place it somewhere on your machine, e.g.:

```
/your/local/path/force-insertion-data/rect_small_tight_200hz_2026-03-17/
    ├── dataset_index.yaml
    ├── episode_0000/episode.h5
    ├── episode_0001/episode.h5
    └── ...
```

### 3. Configure your data path

```bash
cp configs/data_config.example.yaml configs/data_config.yaml
```

Then open `configs/data_config.yaml` and set your local path:

```yaml
data_dir: /your/local/path/force-insertion-data/rect_small_tight_200hz_2026-03-17
success_only: true
val_ratio: 0.2
val_seed: 42
```

`data_config.yaml` is `.gitignored` — everyone sets their own path locally, never commit it.

### 4. Compute normalization stats

Run this once per dataset version. It computes per-channel mean/std over the training split and writes to `configs/normalization_stats.yaml`:

```bash
python scripts/compute_normalization.py
```

`normalization_stats.yaml` **is** tracked in git, so you only need to rerun this if the dataset changes.

---

## Project Structure

```
force-insertion-policy/
├── configs/
│   ├── data_config.example.yaml   ← copy this, never commit data_config.yaml
│   ├── data_config.yaml           ← your local data path (gitignored)
│   └── normalization_stats.yaml   ← tracked, recompute if dataset changes
├── diffusion/                     ← diffusion policy model
├── cvae/                          ← CVAE model
├── shared/
│   └── normalization.py           ← normalize / denormalize utilities
├── scripts/
│   └── compute_normalization.py   ← run once per dataset version
├── dataset/                       ← symlink or copy your data here (gitignored)
└── checkpoints/                   ← saved model weights (gitignored)
```

---

## Observation & Action Space

Both models consume the same input/output interface:

| | Signals | Dim |
|--|---------|-----|
| **Observation** | `f_ext` (3) + `f_internal` (6) + `ee_velocity` (6) | 18 |
| **Action** | `Fff` — 6D feed-forward wrench [N, Nm] | 6 |

All channels are normalized to zero mean / unit variance using `configs/normalization_stats.yaml` before being passed to any model.

---

## Models

### Diffusion Policy

> 🚧 In progress

Will be based on the TacDiffusion framework from MIRMI (Haddadin group, TU Munich):

> Wu et al., *TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation*, 2024.

### CVAE

> 🚧 Not yet started

---

## Notes

- All quaternions are in **wxyz** convention throughout (matches force-insertion-sim)
- Timestamps in HDF5 files are simulation time in seconds, starting from 0 per episode
- `val_seed: 42` ensures everyone gets the same train/val split regardless of machine

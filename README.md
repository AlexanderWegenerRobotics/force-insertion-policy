# force-insertion-policy

Diffusion and CVAE policies for force-domain robotic peg-in-hole insertion.

## Setup

```bash
pip install -e .
```

## Data config

Copy the example config and set your local dataset path:

```bash
cp configs/data_config.example.yaml configs/data_config.yaml
```

Edit `data_config.yaml` to point at your local copy of the dataset (downloaded from shared Google Drive). This file is `.gitignored` — each person maintains their own.

## Compute normalization

Run once per dataset version. Output is checked into Git so both teams use identical stats.

```bash
python -m scripts.compute_normalization --config configs/data_config.yaml
```
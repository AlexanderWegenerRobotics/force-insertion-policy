"""Run once per dataset version: computes norm stats from training split and saves to configs/."""
import argparse
from pathlib import Path
import yaml

from shared.normalization import compute_normalization_stats, save_normalization_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--output", default="configs/normalization_stats.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    stats = compute_normalization_stats(cfg["data_dir"], cfg.get("success_only", True))

    save_normalization_stats(stats, args.output)
    print(f"Saved stats from {stats['n_episodes']} episodes ({stats['n_timesteps']} timesteps) -> {args.output}")


if __name__ == "__main__":
    main()

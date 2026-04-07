import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from cvae.model import ConditionalVAE
from shared.insertion_dataset import InsertionDataset


CHANNEL_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def load_model(checkpoint_dir: Path, device: torch.device):
    with open(checkpoint_dir / "config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    model = ConditionalVAE(
        hidden_dim=train_cfg.get("hidden_dim", 512),
        latent_dim=train_cfg.get("latent_dim", 16),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device))
    model.eval()
    return model, train_cfg


def channel_metrics(pred: np.ndarray, target: np.ndarray):
    diff = pred - target
    mse = np.mean(diff ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff), axis=0)
    target_mean = np.mean(target, axis=0, keepdims=True)
    sst = np.sum((target - target_mean) ** 2, axis=0)
    sse = np.sum(diff ** 2, axis=0)
    r2 = 1.0 - (sse / np.maximum(sst, 1e-12))

    corr = []
    for i in range(target.shape[1]):
        pred_i = pred[:, i]
        target_i = target[:, i]
        if np.std(pred_i) < 1e-12 or np.std(target_i) < 1e-12:
            corr.append(0.0)
        else:
            corr.append(float(np.corrcoef(pred_i, target_i)[0, 1]))

    return {
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "r2": r2.tolist(),
        "corr": corr,
    }


def summarize_metrics(name: str, metrics: dict):
    print(f"\n{name}")
    print("channel, rmse, mae, r2, corr")
    for channel, rmse, mae, r2, corr in zip(
        CHANNEL_NAMES,
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
        metrics["corr"],
    ):
        print(f"{channel}, {rmse:.6f}, {mae:.6f}, {r2:.4f}, {corr:.4f}")


def save_pointwise_csv(
    output_path: Path,
    target: np.ndarray,
    posterior_mean: np.ndarray,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    max_rows: int,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestep"]
    for prefix in ("target", "posterior_mean", "prior_mean", "prior_std"):
        header.extend([f"{prefix}_{name}" for name in CHANNEL_NAMES])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        n_rows = min(target.shape[0], max_rows)
        for i in range(n_rows):
            row = [i]
            row.extend(target[i].tolist())
            row.extend(posterior_mean[i].tolist())
            row.extend(prior_mean[i].tolist())
            row.extend(prior_std[i].tolist())
            writer.writerow(row)


def save_metrics_csv(output_path: Path, posterior_metrics: dict, prior_zero_metrics: dict, prior_mean_metrics: dict, avg_prior_std: list[float]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "channel",
                "posterior_rmse",
                "posterior_mae",
                "posterior_r2",
                "posterior_corr",
                "prior_zero_rmse",
                "prior_zero_mae",
                "prior_zero_r2",
                "prior_zero_corr",
                "prior_mean_rmse",
                "prior_mean_mae",
                "prior_mean_r2",
                "prior_mean_corr",
                "avg_prior_std",
            ]
        )
        for i, channel in enumerate(CHANNEL_NAMES):
            writer.writerow(
                [
                    channel,
                    posterior_metrics["rmse"][i],
                    posterior_metrics["mae"][i],
                    posterior_metrics["r2"][i],
                    posterior_metrics["corr"][i],
                    prior_zero_metrics["rmse"][i],
                    prior_zero_metrics["mae"][i],
                    prior_zero_metrics["r2"][i],
                    prior_zero_metrics["corr"][i],
                    prior_mean_metrics["rmse"][i],
                    prior_mean_metrics["mae"][i],
                    prior_mean_metrics["r2"][i],
                    prior_mean_metrics["corr"][i],
                    avg_prior_std[i],
                ]
            )


def save_channel_csvs(output_dir: Path, target: np.ndarray, posterior_mean: np.ndarray, prior_mean: np.ndarray, prior_std: np.ndarray, max_rows: int):
    channels_dir = output_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    n_rows = min(target.shape[0], max_rows)
    for i, channel in enumerate(CHANNEL_NAMES):
        with open(channels_dir / f"{channel}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "target", "posterior_mean", "prior_mean", "prior_std"])
            for t in range(n_rows):
                writer.writerow([t, target[t, i], posterior_mean[t, i], prior_mean[t, i], prior_std[t, i]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--checkpoint_dir", default="checkpoints/cvae")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_prior_samples", type=int, default=8)
    parser.add_argument("--max_export_rows", type=int, default=20000)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    model, train_cfg = load_model(checkpoint_dir, device)

    ds = InsertionDataset(cfg, split=args.split)
    o_prev = ds.obs_prev.to(device)
    o_curr = ds.obs_curr.to(device)
    action = ds.act.to(device)

    with torch.no_grad():
        mu, _ = model.encode(o_prev, o_curr, action)
        posterior_mean_norm = model.decode(o_prev, o_curr, mu)

        zero_latent = torch.zeros_like(mu)
        prior_zero_norm = model.decode(o_prev, o_curr, zero_latent)

        prior_samples = []
        for _ in range(args.num_prior_samples):
            sample_norm = model.sample(o_prev, o_curr)
            prior_samples.append(sample_norm.unsqueeze(0))
        prior_samples_norm = torch.cat(prior_samples, dim=0)
        prior_mean_norm = prior_samples_norm.mean(dim=0)
        prior_std_norm = prior_samples_norm.std(dim=0)

    target = ds.denormalize_action(action.cpu().numpy())
    posterior_mean = ds.denormalize_action(posterior_mean_norm.cpu().numpy())
    prior_zero = ds.denormalize_action(prior_zero_norm.cpu().numpy())
    prior_mean = ds.denormalize_action(prior_mean_norm.cpu().numpy())

    act_std = ds.act_std.astype(np.float32)
    prior_std = prior_std_norm.cpu().numpy() * act_std[None, :]

    posterior_metrics = channel_metrics(posterior_mean, target)
    prior_zero_metrics = channel_metrics(prior_zero, target)
    prior_mean_metrics = channel_metrics(prior_mean, target)

    summarize_metrics("Posterior reconstruction (upper bound)", posterior_metrics)
    summarize_metrics("Prior prediction with z=0 (deployable deterministic signal)", prior_zero_metrics)
    summarize_metrics("Prior prediction mean over samples", prior_mean_metrics)

    avg_prior_std = prior_std.mean(axis=0).tolist()
    print("\nAverage prior sample std by channel")
    for channel, std in zip(CHANNEL_NAMES, avg_prior_std):
        print(f"{channel}, {std:.6f}")

    results = {
        "split": args.split,
        "checkpoint_dir": str(checkpoint_dir),
        "hidden_dim": train_cfg.get("hidden_dim"),
        "latent_dim": train_cfg.get("latent_dim"),
        "num_prior_samples": args.num_prior_samples,
        "posterior_metrics": posterior_metrics,
        "prior_zero_metrics": prior_zero_metrics,
        "prior_mean_metrics": prior_mean_metrics,
        "avg_prior_std": avg_prior_std,
    }

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_metrics_csv(
        output_dir / "metrics_by_channel.csv",
        posterior_metrics=posterior_metrics,
        prior_zero_metrics=prior_zero_metrics,
        prior_mean_metrics=prior_mean_metrics,
        avg_prior_std=avg_prior_std,
    )

    save_pointwise_csv(
        output_dir / "pointwise_comparison.csv",
        target=target,
        posterior_mean=posterior_mean,
        prior_mean=prior_mean,
        prior_std=prior_std,
        max_rows=args.max_export_rows,
    )
    save_channel_csvs(
        output_dir,
        target=target,
        posterior_mean=posterior_mean,
        prior_mean=prior_mean,
        prior_std=prior_std,
        max_rows=args.max_export_rows,
    )

    with open(output_dir / "readme.txt", "w", encoding="utf-8") as f:
        f.write(
            "posterior_mean uses the true action through the encoder and is an upper bound.\n"
            "prior_mean and prior_std come from sampling z ~ N(0, I), which reflects inference-time uncertainty.\n"
            "prior_zero is summarized in summary.json and corresponds to decoding with z = 0.\n"
            f"pointwise_comparison.csv contains the first {min(len(target), args.max_export_rows)} rows only.\n"
            "metrics_by_channel.csv contains one row per action channel.\n"
            "channels/*.csv contains per-channel timestep data.\n"
        )


if __name__ == "__main__":
    main()

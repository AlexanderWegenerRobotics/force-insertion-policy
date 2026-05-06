import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader

from transformer.dataset import SequenceInsertionDataset
from transformer.model import build_transformer_model


CHANNEL_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but CUDA is not available.")
    return torch.device(device_arg)


def load_model(checkpoint_dir: Path, device: torch.device):
    with open(checkpoint_dir / "config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)
    model = build_transformer_model(train_cfg).to(device)
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
        if np.std(pred[:, i]) < 1e-12 or np.std(target[:, i]) < 1e-12:
            corr.append(0.0)
        else:
            corr.append(float(np.corrcoef(pred[:, i], target[:, i])[0, 1]))
    return {
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "r2": r2.tolist(),
        "corr": corr,
    }


def save_metrics_csv(path: Path, metrics: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "rmse", "mae", "r2", "corr"])
        for i, channel in enumerate(CHANNEL_NAMES):
            writer.writerow(
                [
                    channel,
                    metrics["rmse"][i],
                    metrics["mae"][i],
                    metrics["r2"][i],
                    metrics["corr"][i],
                ]
            )


def save_pointwise_csv(path: Path, pred: np.ndarray, target: np.ndarray, max_rows: int):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["timestep"]
        header.extend([f"pred_{name}" for name in CHANNEL_NAMES])
        header.extend([f"target_{name}" for name in CHANNEL_NAMES])
        writer.writerow(header)
        for i in range(min(max_rows, len(target))):
            writer.writerow([i, *pred[i].tolist(), *target[i].tolist()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--checkpoint_dir", default="checkpoints/transformer")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_export_rows", type=int, default=20000)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    model, train_cfg = load_model(checkpoint_dir, device)
    cfg.update(
        {
            "seq_len": train_cfg.get("seq_len", cfg.get("seq_len", 20)),
            "include_prev_action": train_cfg.get("include_prev_action", True),
        }
    )

    ds = SequenceInsertionDataset(cfg, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    preds = []
    targets = []
    with torch.no_grad():
        for seq, action in loader:
            pred = model(seq.to(device)).cpu().numpy()
            preds.append(pred)
            targets.append(action.numpy())

    pred_norm = np.concatenate(preds, axis=0)
    target_norm = np.concatenate(targets, axis=0)
    pred = ds.denormalize_action(pred_norm)
    target = ds.denormalize_action(target_norm)
    metrics = channel_metrics(pred, target)

    print("\nTransformer open-loop evaluation")
    print(f"split: {args.split}")
    print(f"checkpoint: {checkpoint_dir}")
    print("channel, rmse, mae, r2, corr")
    for channel, rmse, mae, r2, corr in zip(
        CHANNEL_NAMES,
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
        metrics["corr"],
    ):
        print(f"{channel}, {rmse:.6f}, {mae:.6f}, {r2:.4f}, {corr:.4f}")

    scalar = {
        "mse": float(np.mean((pred - target) ** 2)),
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
        "mae": float(np.mean(np.abs(pred - target))),
    }
    print(f"\naggregate_rmse={scalar['rmse']:.6f}, aggregate_mae={scalar['mae']:.6f}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluation_type": "open_loop_transformer",
                "split": args.split,
                "checkpoint_dir": str(checkpoint_dir),
                "model_type": train_cfg.get("model_type", "gru"),
                "hidden_dim": train_cfg.get("hidden_dim"),
                "num_layers": train_cfg.get("num_layers"),
                "num_heads": train_cfg.get("num_heads"),
                "seq_len": train_cfg.get("seq_len"),
                "metrics": metrics,
                "scalar_metrics": scalar,
            },
            f,
            indent=2,
        )
    save_metrics_csv(output_dir / "metrics_by_channel.csv", metrics)
    save_pointwise_csv(output_dir / "pointwise_comparison.csv", pred, target, args.max_export_rows)


if __name__ == "__main__":
    main()

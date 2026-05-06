import argparse
import json
import os
import signal
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import yaml
from torch.utils.data import DataLoader

from transformer.dataset import SequenceInsertionDataset
from transformer.model import build_transformer_model


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def weighted_smooth_l1(pred, target, weights):
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).mean()


def evaluate(model, loader, device, channel_weights, use_amp):
    model.eval()
    total_loss = 0.0
    total_sq = torch.zeros(6, device=device)
    total_abs = torch.zeros(6, device=device)
    n_items = 0
    n_batches = 0

    with torch.no_grad():
        for seq, action in loader:
            seq = seq.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(seq)
                loss = weighted_smooth_l1(pred, action, channel_weights)
            diff = pred.float() - action.float()
            total_sq += torch.sum(diff.pow(2), dim=0)
            total_abs += torch.sum(diff.abs(), dim=0)
            n_items += action.shape[0]
            total_loss += loss.item()
            n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "rmse_norm": torch.sqrt(total_sq / max(1, n_items)).cpu().tolist(),
        "mae_norm": (total_abs / max(1, n_items)).cpu().tolist(),
    }


def evaluate_preloaded(model, seq, action, batch_size, channel_weights, use_amp):
    model.eval()
    total_loss = 0.0
    total_sq = torch.zeros(6, device=seq.device)
    total_abs = torch.zeros(6, device=seq.device)
    n_items = 0
    n_batches = 0

    with torch.no_grad():
        for start in range(0, len(action), batch_size):
            end = min(start + batch_size, len(action))
            batch_seq = seq[start:end]
            batch_action = action[start:end]
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(batch_seq)
                loss = weighted_smooth_l1(pred, batch_action, channel_weights)
            diff = pred.float() - batch_action.float()
            total_sq += torch.sum(diff.pow(2), dim=0)
            total_abs += torch.sum(diff.abs(), dim=0)
            n_items += batch_action.shape[0]
            total_loss += loss.item()
            n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "rmse_norm": torch.sqrt(total_sq / max(1, n_items)).cpu().tolist(),
        "mae_norm": (total_abs / max(1, n_items)).cpu().tolist(),
    }


def train(cfg):
    device = resolve_device(cfg.get("device", "auto"))
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_ds = SequenceInsertionDataset(cfg, split="train")
    val_ds = SequenceInsertionDataset(cfg, split="val")
    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows, seq_len: {train_ds.seq_len}")

    batch_size = int(cfg.get("batch_size", 2048))
    num_workers = int(cfg.get("num_workers", 0))
    preload_to_device = bool(cfg.get("preload_to_device", False))
    if preload_to_device:
        if device.type != "cuda":
            raise ValueError("--preload_to_device is intended for CUDA training. Use --device cuda.")
        print("Materializing temporal windows onto GPU. This uses more VRAM but removes DataLoader CPU slicing.")
        train_seq, train_act = train_ds.materialize_windows(device=device)
        val_seq, val_act = val_ds.materialize_windows(device=device)
        print(
            f"Preloaded train: seq={tuple(train_seq.shape)} {train_seq.device}, "
            f"action={tuple(train_act.shape)} {train_act.device}"
        )
        print(
            f"Preloaded val: seq={tuple(val_seq.shape)} {val_seq.device}, "
            f"action={tuple(val_act.shape)} {val_act.device}"
        )
        train_loader = None
        val_loader = None
    else:
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    model = build_transformer_model(cfg).to(device)
    print(f"Model parameter device: {next(model.parameters()).device}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    channel_weights = torch.tensor(
        cfg.get("channel_weights", [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]),
        dtype=torch.float32,
        device=device,
    )
    grad_clip = float(cfg.get("grad_clip", 1.0))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model params: {n_params:,} | type: {cfg.get('model_type', 'gru')} "
        f"| hidden_dim: {cfg.get('hidden_dim', 256)} | layers: {cfg.get('num_layers', 2)}"
    )

    epochs = int(cfg.get("epochs", 500))
    val_every = int(cfg.get("val_every", 5))
    save_dir = Path(cfg.get("save_dir", "checkpoints/transformer"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_epochs": [],
        "val_rmse_norm": [],
        "val_mae_norm": [],
    }
    epoch = 0

    def persist_history():
        history["best_val_loss"] = best_val_loss
        history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None
        history["n_params"] = n_params
        history["model_type"] = cfg.get("model_type", "gru")
        history["hidden_dim"] = int(cfg.get("hidden_dim", 256))
        history["num_layers"] = int(cfg.get("num_layers", 2))
        history["num_heads"] = int(cfg.get("num_heads", 4))
        history["seq_len"] = int(cfg.get("seq_len", 20))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def handle_interrupt(signum, frame):
        print(f"\nInterrupted at epoch {epoch}. Saving checkpoint...")
        torch.save(model.state_dict(), save_dir / "interrupted.pt")
        persist_history()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    for epoch in range(1, epochs + 1):
        epoch_time = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        if preload_to_device:
            perm = torch.randperm(len(train_act), device=device)
            for start in range(0, len(train_act), batch_size):
                idx = perm[start:start + batch_size]
                seq = train_seq[idx]
                action = train_act[idx]
                if n_batches == 0 and epoch == 1:
                    print(f"First preloaded batch: seq={seq.device}, action={action.device}")

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(seq)
                    loss = weighted_smooth_l1(pred, action, channel_weights)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                n_batches += 1
        else:
            for seq, action in train_loader:
                if n_batches == 0 and epoch == 1:
                    print(f"First batch before transfer: seq={seq.device}, action={action.device}")
                seq = seq.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)
                if n_batches == 0 and epoch == 1:
                    print(f"First batch after transfer: seq={seq.device}, action={action.device}")

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(seq)
                    loss = weighted_smooth_l1(pred, action, channel_weights)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        history["train_loss"].append(train_loss)

        if epoch % val_every == 0:
            if preload_to_device:
                val_metrics = evaluate_preloaded(
                    model,
                    val_seq,
                    val_act,
                    batch_size,
                    channel_weights,
                    use_amp,
                )
            else:
                val_metrics = evaluate(model, val_loader, device, channel_weights, use_amp)
            val_loss = val_metrics["loss"]
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)
            history["val_rmse_norm"].append(val_metrics["rmse_norm"])
            history["val_mae_norm"].append(val_metrics["mae_norm"])

            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "best.pt")
                improved = " *"

            print(
                f"Epoch {epoch:4d} | {time.time() - epoch_time:.2f}s | "
                f"train: {train_loss:.5f} | val: {val_loss:.5f}{improved}"
            )
            persist_history()
        else:
            print(f"Epoch {epoch:4d} | {time.time() - epoch_time:.2f}s | train: {train_loss:.5f}")

    torch.save(model.state_dict(), save_dir / "final.pt")
    persist_history()
    print(f"Done. Best val loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--model_type", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=None)
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", default="checkpoints/transformer")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--preload_to_device", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    cfg.setdefault("include_prev_action", True)
    cfg.setdefault("channel_weights", [1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    train(cfg)

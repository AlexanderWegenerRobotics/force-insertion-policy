import argparse
import json
import os
import signal
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import yaml

from cvae.model import ConditionalVAE
from shared.insertion_dataset import InsertionDataset


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and make sure your NVIDIA driver is available."
        )
    return torch.device(device_arg)


def kl_weight_for_epoch(epoch: int, beta: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta
    return beta * min(epoch / warmup_epochs, 1.0)


def evaluate(model, o_prev, o_curr, action, batch_size, kl_weight, use_amp):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    with torch.no_grad():
        for start in range(0, len(action), batch_size):
            end = min(start + batch_size, len(action))
            batch_prev = o_prev[start:end]
            batch_curr = o_curr[start:end]
            batch_act = action[start:end]

            with torch.amp.autocast("cuda", enabled=use_amp):
                action_pred, mu, logvar = model(batch_prev, batch_curr, batch_act)
                recon_loss = torch.nn.functional.mse_loss(action_pred, batch_act)
                kl_loss = model.kl_divergence(mu, logvar)
                loss = recon_loss + kl_weight * kl_loss

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


def train(cfg):
    device = resolve_device(cfg.get("device", "auto"))
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = InsertionDataset(cfg, split="train")
    val_ds = InsertionDataset(cfg, split="val")
    print(f"Train: {len(train_ds)} timesteps, Val: {len(val_ds)} timesteps")

    train_o_prev = train_ds.obs_prev.to(device)
    train_o_curr = train_ds.obs_curr.to(device)
    train_act = train_ds.act.to(device)

    val_o_prev = val_ds.obs_prev.to(device)
    val_o_curr = val_ds.obs_curr.to(device)
    val_act = val_ds.act.to(device)

    n_train = len(train_ds)
    batch_size = cfg.get("batch_size", 4096)
    if device.type == "cuda":
        print(f"GPU memory after data load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model = ConditionalVAE(
        hidden_dim=cfg.get("hidden_dim", 512),
        latent_dim=cfg.get("latent_dim", 16),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    n_params = sum(p.numel() for p in model.parameters())
    beta = cfg.get("beta", 1e-3)
    kl_warmup_epochs = cfg.get("kl_warmup_epochs", 100)
    print(
        f"Model params: {n_params:,} | hidden_dim: {cfg.get('hidden_dim', 512)} "
        f"| latent_dim: {cfg.get('latent_dim', 16)} | beta: {beta}"
    )

    epochs = cfg.get("epochs", 1500)
    val_every = cfg.get("val_every", 5)
    best_val_loss = float("inf")
    save_dir = Path(cfg.get("save_dir", "checkpoints/cvae"))
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    history = {
        "train_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "train_kl_weight": [],
        "val_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
        "val_epochs": [],
    }
    epoch = 0

    def persist_history():
        history["best_val_loss"] = best_val_loss
        history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None
        history["n_params"] = n_params
        history["hidden_dim"] = cfg.get("hidden_dim", 512)
        history["latent_dim"] = cfg.get("latent_dim", 16)
        history["beta"] = beta
        history["kl_warmup_epochs"] = kl_warmup_epochs
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def handle_interrupt(signum, frame):
        print(f"\nInterrupted at epoch {epoch}. Saving checkpoint...")
        torch.save(model.state_dict(), save_dir / "interrupted.pt")
        persist_history()
        print(f"Saved to {save_dir}. Best val loss: {best_val_loss:.4f}")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        epoch_time = time.time()
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0
        kl_weight = kl_weight_for_epoch(epoch, beta, kl_warmup_epochs)

        perm = torch.randperm(n_train, device=device)

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            o_prev = train_o_prev[idx]
            o_curr = train_o_curr[idx]
            action = train_act[idx]

            with torch.amp.autocast("cuda", enabled=use_amp):
                action_pred, mu, logvar = model(o_prev, o_curr, action)
                recon_loss = torch.nn.functional.mse_loss(action_pred, action)
                kl_loss = model.kl_divergence(mu, logvar)
                loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        train_recon = total_recon / n_batches
        train_kl = total_kl / n_batches
        history["train_loss"].append(train_loss)
        history["train_recon_loss"].append(train_recon)
        history["train_kl_loss"].append(train_kl)
        history["train_kl_weight"].append(kl_weight)

        if epoch % val_every == 0:
            val_metrics = evaluate(
                model,
                val_o_prev,
                val_o_curr,
                val_act,
                batch_size=batch_size,
                kl_weight=kl_weight,
                use_amp=use_amp,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_recon_loss"].append(val_metrics["recon_loss"])
            history["val_kl_loss"].append(val_metrics["kl_loss"])
            history["val_epochs"].append(epoch)

            improved = ""
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), save_dir / "best.pt")
                improved = " *"

            print(
                f"Epoch {epoch:4d} | {time.time() - epoch_time:.2f}s | "
                f"train: {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}, w {kl_weight:.4f}) | "
                f"val: {val_metrics['loss']:.4f} (recon {val_metrics['recon_loss']:.4f}, kl {val_metrics['kl_loss']:.4f})"
                f"{improved}"
            )
            persist_history()
        else:
            print(
                f"Epoch {epoch:4d} | {time.time() - epoch_time:.2f}s | "
                f"train: {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}, w {kl_weight:.4f})"
            )

    torch.save(model.state_dict(), save_dir / "final.pt")
    persist_history()
    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--kl_warmup_epochs", type=int, default=100)
    parser.add_argument("--save_dir", default="checkpoints/cvae")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    train(cfg)

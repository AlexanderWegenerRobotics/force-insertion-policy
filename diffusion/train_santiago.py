import torch
import yaml
import argparse
import json
import signal
import time
from pathlib import Path

from shared.insertion_dataset import InsertionDataset
from diffusion.ddpm_santiago import build_noise_estimator, DDPMSchedule

MODEL_CHOICES = [
    "baseline",
    "simple_mlp",
    "compact_width",
    "shallow_backbone",
    "shared_observation",
    "fused_observation",
    "fused_observation_large",
]


def measure_inference_time(model, cfg, device, n_runs=200):
    """Single-sample forward pass latency in milliseconds (batch_size=1)."""
    model.eval()
    obs_dim    = cfg.get("obs_dim", 18)
    action_dim = cfg.get("action_dim", 6)
    o_prev  = torch.randn(1, obs_dim,    device=device)
    o_curr  = torch.randn(1, obs_dim,    device=device)
    action  = torch.randn(1, action_dim, device=device)
    tau     = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(20):           # warmup
            model(o_prev, o_curr, action, tau)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(o_prev, o_curr, action, tau)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000


def save_run(save_dir, model, cfg, history, n_params, suffix=""):
    save_dir = Path(save_dir)
    torch.save(model.state_dict(), save_dir / "latest.pt")  # always keep latest

    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    arch = {
        "class": model.__class__.__name__,
        "variant": cfg.get("model", "baseline"),          # added
        "str": str(model),
        "n_params": n_params,
        "obs_dim": cfg.get("obs_dim", 18),
        "action_dim": cfg.get("action_dim", 6),
        "hidden_dim": cfg.get("hidden_dim", 512),
        "timestep_embed_dim": cfg.get("timestep_embed_dim", 128),
        "diffusion_horizon": cfg.get("diffusion_horizon", 50),
        "noise_schedule": {
            "type": "linear",
            "beta_start": cfg.get("beta_start", 1e-4),
            "beta_end": cfg.get("beta_end", 1e-2),
            "T": cfg.get("diffusion_horizon", 50),
        },
    }
    with open(save_dir / "architecture.json", "w") as f:
        json.dump(arch, f, indent=2)

    if suffix:
        torch.save(model.state_dict(), save_dir / f"{suffix}.pt")


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = InsertionDataset(cfg, split="train")
    val_ds = InsertionDataset(cfg, split="val")
    print(f"Train: {len(train_ds)} timesteps, Val: {len(val_ds)} timesteps")

    train_o_prev = train_ds.obs_prev.to(device)
    train_o_curr = train_ds.obs_curr.to(device)
    train_act = train_ds.act.to(device)

    val_o_prev = val_ds.obs_prev.to(device)
    val_o_curr = val_ds.obs_curr.to(device)
    val_act = val_ds.act.to(device)

    N_train = len(train_ds)
    N_val = len(val_ds)
    batch_size = cfg.get("batch_size", 4096)

    if device.type == "cuda":
        print(f"GPU memory after data load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    T = cfg.get("diffusion_horizon", 50)

    # changed: model selected by variant name; only pass hidden_dim for baseline
    model_name = cfg.get("model", "baseline")
    model_kwargs = {"obs_dim": cfg.get("obs_dim", 18), "action_dim": cfg.get("action_dim", 6)}
    if model_name == "baseline":
        model_kwargs["hidden_dim"] = cfg.get("hidden_dim", 512)
    model = build_noise_estimator(model_name, **model_kwargs).to(device)

    # Infer actual hidden_dim from model for accurate logging
    if hasattr(model, 'residual_blocks') and len(model.residual_blocks) > 0:
        cfg['hidden_dim'] = model.residual_blocks[0].net[0].out_features
    elif hasattr(model, 'net'):
        cfg['hidden_dim'] = model.net[0].out_features

    schedule = DDPMSchedule(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {n_params:,} | T: {T}")

    epochs = cfg.get("epochs", 1500)
    best_val_loss = float("inf")
    best_epoch = 0

    patience = cfg.get("early_stopping_patience", 30)
    min_delta = cfg.get("early_stopping_min_delta", 3e-4)
    epochs_without_improvement = 0

    save_dir = Path(cfg.get("save_dir", f"checkpoints/{model_name}"))
    save_dir.mkdir(exist_ok=True, parents=True)
    val_every = cfg.get("val_every", 5)

    history = {
        "model": model_name,                              # added
        "train_loss": [],
        "val_loss": [],
        "val_epochs": [],
        "best_val_loss": None,
        "best_epoch": None,
        "final_train_loss": None,
        "n_params": n_params,
        "hidden_dim": cfg.get("hidden_dim", 512),
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
    }
    epoch = 0

    def handle_interrupt(signum, frame):
        print(f"\nInterrupted at epoch {epoch}. Saving checkpoint...")
        history["best_val_loss"] = best_val_loss
        history["best_epoch"] = best_epoch
        history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None
        history["interrupted_epoch"] = epoch
        save_run(save_dir, model, cfg, history, n_params, suffix="interrupted")
        print(f"Saved to {save_dir}. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        epoch_time = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        perm = torch.randperm(N_train, device=device)

        for start in range(0, N_train - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            o_prev = train_o_prev[idx]
            o_curr = train_o_curr[idx]
            action = train_act[idx]

            tau = torch.randint(0, T, (batch_size,), device=device)
            a_noised, noise = schedule.q_sample(action, tau)

            with torch.amp.autocast("cuda", enabled=use_amp):
                noise_pred = model(o_prev, o_curr, a_noised, tau)
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches were processed. Check batch_size and dataset size.")

        train_loss = total_loss / n_batches
        history["train_loss"].append(train_loss)

        if epoch % val_every == 0:
            model.eval()
            val_total = 0.0
            val_batches = 0

            with torch.no_grad():
                for start in range(0, N_val, batch_size):
                    end = min(start + batch_size, N_val)
                    o_prev = val_o_prev[start:end]
                    o_curr = val_o_curr[start:end]
                    action = val_act[start:end]

                    tau = torch.randint(0, T, (end - start,), device=device)
                    a_noised, noise = schedule.q_sample(action, tau)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        noise_pred = model(o_prev, o_curr, a_noised, tau)
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    val_total += loss.item()
                    val_batches += 1

            val_loss = val_total / val_batches
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)
            history["final_train_loss"] = train_loss

            improved = ""
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), save_dir / "best.pt")
                improved = " *"
            else:
                epochs_without_improvement += 1

            history["best_val_loss"] = best_val_loss
            history["best_epoch"] = best_epoch

            save_run(save_dir, model, cfg, history, n_params)

            print(
                f"Epoch {epoch:4d} | {time.time() - epoch_time:.2f}s | "
                f"train: {train_loss:.6f} | val: {val_loss:.6f}{improved} | "
                f"bad val checks: {epochs_without_improvement}/{patience}"
            )

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}."
                )
                break
        else:
            print(f"Epoch {epoch:4d} | {time.time() - epoch_time:.2f}s | train: {train_loss:.6f}")

    history["best_val_loss"] = best_val_loss
    history["best_epoch"] = best_epoch
    history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None

    torch.save(model.state_dict(), save_dir / "final.pt")

    inference_ms = measure_inference_time(model, cfg, device)
    history["inference_time_ms"] = round(inference_ms, 4)
    print(f"Inference time (single sample): {inference_ms:.4f} ms")

    save_run(save_dir, model, cfg, history, n_params)
    print(f"Done. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--model", default="baseline", choices=MODEL_CHOICES)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default=None)   # changed: None so model name is auto-used
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--early_stopping_min_delta", type=float, default=3e-4)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    train(cfg)

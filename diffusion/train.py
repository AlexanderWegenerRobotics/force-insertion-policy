import torch
import yaml
import argparse
import json
import signal
import time
from pathlib import Path

from shared.insertion_dataset import InsertionDataset
from diffusion.ddpm import NoiseEstimator, DDPMSchedule


def save_run(save_dir, model, cfg, history, n_params, suffix=""):
    save_dir = Path(save_dir)

    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    arch = {
        "class": model.__class__.__name__,
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
    print(f"GPU memory after data load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    T = cfg.get("diffusion_horizon", 50)
    model = NoiseEstimator(hidden_dim=cfg.get("hidden_dim", 512)).to(device)
    schedule = DDPMSchedule(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} | hidden_dim: {cfg.get('hidden_dim', 512)} | T: {T}")

    epochs = cfg.get("epochs", 1500)
    best_val_loss = float("inf")
    save_dir = Path(cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(exist_ok=True, parents=True)
    val_every = cfg.get("val_every", 5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_epochs": [],
        "best_val_loss": None,
        "final_train_loss": None,
        "n_params": n_params,
        "hidden_dim": cfg.get("hidden_dim", 512),
    }
    epoch = 0

    def handle_interrupt(signum, frame):
        print(f"\nInterrupted at epoch {epoch}. Saving checkpoint...")
        history["best_val_loss"] = best_val_loss
        history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None
        history["interrupted_epoch"] = epoch
        save_run(save_dir, model, cfg, history, n_params, suffix="interrupted")
        print(f"Saved to {save_dir}. Best val loss: {best_val_loss:.4f}")
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda")
    use_amp = device.type == "cuda"

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
                        val_total += torch.nn.functional.mse_loss(noise_pred, noise).item()
                    val_batches += 1

            val_loss = val_total / val_batches
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)
            history["best_val_loss"] = best_val_loss
            history["final_train_loss"] = train_loss

            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "best.pt")
                improved = " *"

            save_run(save_dir, model, cfg, history, n_params)

            print(f"Epoch {epoch:4d} | {time.time() - epoch_time:.2f}s | train: {train_loss:.4f} | val: {val_loss:.4f}{improved}")
        else:
            print(f"Epoch {epoch:4d} | {time.time() - epoch_time:.1f}s | train: {train_loss:.4f}")

    history["best_val_loss"] = best_val_loss
    history["final_train_loss"] = history["train_loss"][-1]
    torch.save(model.state_dict(), save_dir / "final.pt")
    save_run(save_dir, model, cfg, history, n_params)
    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    train(cfg)
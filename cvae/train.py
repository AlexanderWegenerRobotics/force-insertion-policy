import torch
import yaml
import argparse
import json
import signal
import time
from pathlib import Path

from shared.insertion_dataset import InsertionDataset
from cvae.cVAE.cvae import CVAE, loss_function_MSE

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

    action_dim = train_act.shape[-1]
    single_obs_dim = train_o_curr.shape[-1]

    N_train = len(train_ds)
    N_val = len(val_ds)
    batch_size = cfg.get("batch_size", 4096)
    print(f"GPU memory after data load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    T = cfg.get("diffusion_horizon", 50)
#    model = NoiseEstimator(latent_size=cfg.get("latent_size", 512)).to(device)
    model = CVAE(action_dim, cfg.get("latent_size",20), single_obs_dim*2).to(device)
#    schedule = DDPMSchedule(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} | latent_size: {cfg.get('latent_size', 20)} | T: {T}")

    epochs = cfg.get("epochs", 1500)
    best_val_loss = float("inf")
    save_dir = Path(cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    val_every = cfg.get("val_every", 5)

    history = {"train_loss": [], "val_loss": [], "val_epochs": []}
    epoch = 0

    def handle_interrupt(signum, frame):
        print(f"\nInterrupted at epoch {epoch}. Saving checkpoint...")
        torch.save(model.state_dict(), save_dir / "interrupted.pt")
        history["best_val_loss"] = best_val_loss
        history["final_train_loss"] = history["train_loss"][-1] if history["train_loss"] else None
        history["n_params"] = n_params
        history["latent_size"] = cfg.get("latent_size", 20)
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f)
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
            o_both = torch.cat([o_curr, o_prev], dim=-1)
            action = train_act[idx]

            # tau = torch.randint(0, T, (batch_size,), device=device)
            # a_noised, noise = schedule.q_sample(action, tau)

            # with torch.amp.autocast("cuda", enabled=use_amp):
            #     noise_pred = model(o_prev, o_curr, a_noised, tau)
            #     loss = torch.nn.functional.mse_loss(noise_pred, noise)

#            data, labels = data.to(device), labels.to(device)
#            labels = one_hot(labels, 10)

            with torch.amp.autocast("cuda", enabled=use_amp):
                recon_batch, mu, logvar = model(action, o_both)
                loss = loss_function_MSE(recon_batch, action, mu, logvar)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        history["train_loss"].append(train_loss)

        # Validation score 
        if epoch % val_every == 0:
            model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for start in range(0, N_val, batch_size):
                    end = min(start + batch_size, N_val)
                    o_prev = val_o_prev[start:end]
                    o_curr = val_o_curr[start:end]
                    o_both = torch.cat([o_curr, o_prev], dim=-1)
                    action = val_act[start:end]

#                    tau = torch.randint(0, T, (end - start,), device=device)
#                    a_noised, noise = schedule.q_sample(action, tau)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        recon_batch, mu, logvar = model(action, o_both)
                        val_total += loss_function_MSE(recon_batch, action, mu, logvar).item()

                    val_batches += 1

            val_loss = val_total / val_batches
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)

            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "best.pt")
                improved = " *"

            print(f"Epoch {epoch:4d} | {time.time() - epoch_time:.2f}s | train: {train_loss:.4f} | val: {val_loss:.4f}{improved}")
            history["best_val_loss"] = best_val_loss
            history["final_train_loss"] = history["train_loss"][-1]
            history["n_params"] = n_params
            history["latent_size"] = cfg.get("latent_size", 20)
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f)
        else:
            print(f"Epoch {epoch:4d} | {time.time() - epoch_time:.1f}s | train: {train_loss:.4f}")

    torch.save(model.state_dict(), save_dir / "final.pt")
    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    train(cfg)
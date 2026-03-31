import torch
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from shared.insertion_dataset import InsertionDataset
from diffusion.ddpm import NoiseEstimator, DDPMSchedule


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = InsertionDataset(cfg, split="train")
    val_ds = InsertionDataset(cfg, split="val")
    print(f"Train: {len(train_ds)} timesteps, Val: {len(val_ds)} timesteps")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4096), shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4096), shuffle=False, num_workers=4, pin_memory=True)

    T = cfg.get("diffusion_horizon", 50)
    model = NoiseEstimator(hidden_dim=cfg.get("hidden_dim", 512)).to(device)
    schedule = DDPMSchedule(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} | hidden_dim: {cfg.get('hidden_dim', 512)} | T: {T}")

    epochs = cfg.get("epochs", 1500)
    best_val_loss = float("inf")
    save_dir = Path(cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(exist_ok=True)
    val_every = cfg.get("val_every", 5)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for o_prev, o_curr, action in train_loader:
            o_prev, o_curr, action = o_prev.to(device), o_curr.to(device), action.to(device)
            tau = torch.randint(0, T, (action.shape[0],), device=device)
            a_noised, noise = schedule.q_sample(action, tau)
            noise_pred = model(o_prev, o_curr, a_noised, tau)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        if epoch % val_every == 0:
            model.eval()
            val_total = 0
            val_batches = 0
            with torch.no_grad():
                for o_prev, o_curr, action in val_loader:
                    o_prev, o_curr, action = o_prev.to(device), o_curr.to(device), action.to(device)
                    tau = torch.randint(0, T, (action.shape[0],), device=device)
                    a_noised, noise = schedule.q_sample(action, tau)
                    noise_pred = model(o_prev, o_curr, a_noised, tau)
                    val_total += torch.nn.functional.mse_loss(noise_pred, noise).item()
                    val_batches += 1

            val_loss = val_total / val_batches
            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "best.pt")
                improved = " *"

            print(f"Epoch {epoch:4d} | train: {train_loss:.4f} | val: {val_loss:.4f}{improved}")
        else:
            print(f"Epoch {epoch:4d} | train: {train_loss:.4f}")

    torch.save(model.state_dict(), save_dir / "final.pt")
    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    train(cfg)

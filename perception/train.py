import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.environ['PROJECT_PATH'])

from perception.models import OrigUNet
from perception.dataset import EventDepthDataset
from perception.plot import (plot_curves, plot_qualitative,
                              plot_scatter, plot_error_histogram)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATASETS_DIR  = os.path.join(os.environ['PROJECT_PATH'], 'datasets')
CKPT_DIR      = os.path.join(os.environ['PROJECT_PATH'], 'perception', 'checkpoints')
PLOTS_DIR     = os.path.join(os.environ['PROJECT_PATH'], 'perception', 'plots')
EPOCHS        = 200
BATCH_SIZE    = 16
LR            = 1e-4
VAL_SPLIT     = 0.15      # fraction of data held out for validation
DEPTH_THRESH  = 0.99      # ignore pixels with normalised depth > this (background)
WORKERS       = 4
QUAL_EVERY    = 20        # save qualitative grid every N epochs
QUAL_SAMPLES  = 4         # number of val samples to show in the grid
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'


# ──────────────────────────────────────────────
# Loss / metrics
# ──────────────────────────────────────────────
def masked_mse(pred, target, thresh=DEPTH_THRESH):
    """MSE loss masked to valid (non-background) depth pixels."""
    mask  = (target < thresh).float()
    loss  = ((pred - target) ** 2) * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def compute_metrics(pred, target, thresh=DEPTH_THRESH):
    """
    MAE, RMSE (in metres), and δ<1.25 accuracy for valid pixels.

    pred, target : torch tensors with values in [0, 1]
                   metric depth = value × 100 m
    Returns dict with keys 'mae', 'rmse', 'delta1'.
    """
    mask = target < thresh
    if mask.sum() == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'delta1': 0.0}

    p = pred[mask]   * 100.0    # normalised → metres
    t = target[mask] * 100.0

    mae   = (p - t).abs().mean().item()
    rmse  = ((p - t) ** 2).mean().sqrt().item()

    ratio  = torch.max(p / t.clamp(min=1e-3), t.clamp(min=1e-3) / p.clamp(min=1e-3))
    delta1 = (ratio < 1.25).float().mean().item()

    return {'mae': mae, 'rmse': rmse, 'delta1': delta1}


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def run():
    os.makedirs(CKPT_DIR,  exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────
    dataset  = EventDepthDataset(DATASETS_DIR)
    n_val    = max(1, int(len(dataset) * VAL_SPLIT))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, pin_memory=True)

    print(f'Train: {n_train}  Val: {n_val}  Device: {DEVICE}')

    # ── Model ─────────────────────────────────
    model    = OrigUNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    # Tracking history for plots
    epochs_train = []
    losses_train = []
    epochs_val   = []
    losses_val   = []
    metrics_val  = []   # list of dicts {mae, rmse, delta1}

    # ── Epoch loop ────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        train_loss = 0.0
        for event, depth in train_loader:
            event = event.to(DEVICE)
            depth = depth.to(DEVICE)

            pred = model(event)
            loss = masked_mse(pred, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * event.size(0)

        train_loss /= n_train
        epochs_train.append(epoch)
        losses_train.append(train_loss)

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_loss   = 0.0
            sum_mae    = 0.0
            sum_rmse   = 0.0
            sum_d1     = 0.0

            with torch.no_grad():
                for event, depth in val_loader:
                    event = event.to(DEVICE)
                    depth = depth.to(DEVICE)
                    pred  = model(event)

                    n = event.size(0)
                    val_loss += masked_mse(pred, depth).item() * n

                    m = compute_metrics(pred, depth)
                    sum_mae  += m['mae']  * n
                    sum_rmse += m['rmse'] * n
                    sum_d1   += m['delta1'] * n

            val_loss /= n_val
            epoch_metrics = {
                'mae':    sum_mae  / n_val,
                'rmse':   sum_rmse / n_val,
                'delta1': sum_d1   / n_val,
            }

            epochs_val.append(epoch)
            losses_val.append(val_loss)
            metrics_val.append(epoch_metrics)

            print(f"Epoch {epoch:04d}/{EPOCHS}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"MAE={epoch_metrics['mae']:.3f}m  "
                  f"RMSE={epoch_metrics['rmse']:.3f}m  "
                  f"δ<1.25={epoch_metrics['delta1']:.3f}")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(CKPT_DIR, 'best.pth')
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'val_loss': val_loss, 'metrics': epoch_metrics}, ckpt_path)
                print(f"    -> saved best checkpoint (val={val_loss:.6f})")

            # loss + metric curves (updated every val check)
            plot_curves(epochs_train, losses_train,
                        epochs_val,   losses_val, metrics_val, PLOTS_DIR)

            # qualitative grid (every QUAL_EVERY epochs)
            if epoch % QUAL_EVERY == 0 or epoch == 1:
                plot_qualitative(model, val_loader, DEVICE, epoch, PLOTS_DIR,
                                 n_samples=QUAL_SAMPLES, thresh=DEPTH_THRESH)

        else:
            print(f'Epoch {epoch}/{EPOCHS}  train={train_loss:.6f}')

    # ── End-of-training plots ─────────────────
    print("Generating final diagnostic plots …")
    plot_scatter(model, val_loader, DEVICE, PLOTS_DIR, thresh=DEPTH_THRESH)
    plot_error_histogram(model, val_loader, DEVICE, PLOTS_DIR, thresh=DEPTH_THRESH)

    # Save final checkpoint
    torch.save({'epoch': EPOCHS, 'model': model.state_dict()},
               os.path.join(CKPT_DIR, 'final.pth'))
    print(f"Training complete. Plots saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    run()

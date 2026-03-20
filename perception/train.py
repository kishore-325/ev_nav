import os
import sys
import csv
import time
import optuna

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.environ['PROJECT_PATH'])

from perception.models import OrigUNet
from perception.dataset import EventDepthDataset
from perception.plot import (plot_curves, plot_qualitative,
                              plot_scatter, plot_error_histogram)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
TRAIN_DATASETS_DIR = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Train')
VAL_DATASETS_DIR   = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Val')
TEST_DATASETS_DIR  = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Test')
CKPT_DIR           = os.path.join('/home/srinivasan/ev_nav_run2', 'perception', 'checkpoints')
PLOTS_DIR          = os.path.join('/home/srinivasan/ev_nav_run2', 'perception', 'plots')
EPOCHS        = 200
BATCH_SIZE    = 64
EARLY_STOPPING_PATIENCE = 7
DEPTH_THRESH  = 0.20   # ignore pixels with normalised depth > this (background)
WORKERS       = 4
QUAL_EVERY    = 20        # save qualitative grid every N epochs
QUAL_SAMPLES  = 4         # number of val samples to show in the grid
QUAL_SKIP     = 2         # number of batches to skip before sampling the qual grid
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# ──────────────────────────────────────────────
# Optuna (load best params if study exists)
# ──────────────────────────────────────────────
try:
    study = optuna.load_study(study_name='unet_depth', storage='sqlite:///optuna_study.db')
    best = study.best_params
    LR            = best['lr']
    WEIGHT_OFFSET = best['weight_offset']
    BERHU_C_FRAC  = best['berhu_c_frac']
    GRAD_WEIGHT   = best['grad_weight']
    print(f"Loaded Optuna best params: lr={LR:.2e}, weight_offset={WEIGHT_OFFSET:.4f}, "
          f"berhu_c_frac={BERHU_C_FRAC:.4f}, grad_weight={GRAD_WEIGHT:.4f}")
except Exception:
    LR            = 1e-4
    WEIGHT_OFFSET = 0.02
    BERHU_C_FRAC  = 0.2
    GRAD_WEIGHT   = 0.5
    print("No Optuna study found, using default hyperparameters.")

# ──────────────────────────────────────────────
# Loss / metrics
# ──────────────────────────────────────────────
def gradient_loss(pred, target, thresh=DEPTH_THRESH):
    mask = (target >= 0) & (target <= thresh)
    mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = (pred_dx - target_dx).abs()[mask_x].mean()
    loss_y = (pred_dy - target_dy).abs()[mask_y].mean()
    return loss_x + loss_y

def combined_loss(pred, target, thresh=DEPTH_THRESH, weight_offset=0.02, berhu_c_frac=0.2, grad_weight=0.5):
    mask = (target >= 0) & (target <= thresh)
    weight = (1.0 / (target + weight_offset)) * mask.float()
    diff = (pred-target).abs()
    c = berhu_c_frac * diff[mask].max().detach()
    bh = torch.where(diff <= c, diff, (diff**2 + c**2) / (2*c))
    main = (weight * bh).mean()
    grad = gradient_loss(pred, target, thresh)
    return main + grad_weight*grad


def compute_metrics(pred, target, thresh=DEPTH_THRESH):
    """
    MAE, RMSE (in metres), and δ<1.25 accuracy for valid pixels.

    pred, target : torch tensors with values in [0, 1]
                   metric depth = value * 100 m
    Returns dict with keys 'mae', 'rmse', 'delta1'.
    """
    mask = (target >= 0) & (target <= thresh)
    if mask.sum() == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'delta1': 0.0}

    p = pred[mask]   * 100.0
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
    train_dataset = EventDepthDataset(TRAIN_DATASETS_DIR)
    val_dataset   = EventDepthDataset(VAL_DATASETS_DIR)
    test_dataset  = EventDepthDataset(TEST_DATASETS_DIR)
    n_train = len(train_dataset)
    n_val   = len(val_dataset)
    n_test  = len(test_dataset)

    n_gpus = min(torch.cuda.device_count() if torch.cuda.is_available() else 1, 4)
    effective_batch = BATCH_SIZE * n_gpus

    train_loader = DataLoader(train_dataset, batch_size=effective_batch, shuffle=True,
                              num_workers=WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=effective_batch, shuffle=False,
                              num_workers=WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=effective_batch, shuffle=False,
                              num_workers=WORKERS, pin_memory=True)

    print(f'Train: {n_train}  Val: {n_val}  Test: {n_test}  Device: {DEVICE}  GPU: {n_gpus}  BATCH: {effective_batch}')

    # ── Model ─────────────────────────────────
    model    = OrigUNet().to(DEVICE)
    if n_gpus > 1:
        print(f"Using {n_gpus} GPU's")
        model = torch.nn.DataParallel(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    patience_counter = 0

    best_val_loss = float('inf')

    log_path = os.path.join(PLOTS_DIR, 'train_log.csv')
    log_file = open(log_path, 'w', newline='')
    logger = csv.writer(log_file)
    logger.writerow(['epoch', 'train_loss',
                     'val_loss', 'val_mae_m', 'val_rmse_m', 'val_delta1',
                     'test_loss', 'test_mae_m', 'test_rmse_m', 'test_delta1', 'lr'])

    # Tracking history for plots
    epochs_train = []
    losses_train = []
    epochs_val   = []
    losses_val   = []
    metrics_val  = []   # list of dicts {mae, rmse, delta1}
    epochs_test  = []
    losses_test  = []
    metrics_test = []   # list of dicts {mae, rmse, delta1}

    # ── Epoch loop ────────────────────────────
    start = time.time()
    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        train_loss = 0.0
        for event, depth in train_loader:
            event = event.to(DEVICE)
            depth = depth.to(DEVICE)

            pred = model(event)
            loss = combined_loss(pred, depth,
                                 weight_offset=WEIGHT_OFFSET,
                                 berhu_c_frac=BERHU_C_FRAC,
                                 grad_weight=GRAD_WEIGHT)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * event.size(0)

        train_loss /= n_train
        epochs_train.append(epoch)
        losses_train.append(train_loss)

        # Validate and test every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()

            # -- Val --
            val_loss = 0.0
            sum_mae  = 0.0; sum_rmse = 0.0; sum_d1 = 0.0
            with torch.no_grad():
                for event, depth in val_loader:
                    event = event.to(DEVICE); depth = depth.to(DEVICE)
                    pred  = model(event)
                    n = event.size(0)
                    val_loss += combined_loss(pred, depth,
                                 weight_offset=WEIGHT_OFFSET,
                                 berhu_c_frac=BERHU_C_FRAC,
                                 grad_weight=GRAD_WEIGHT).item() * n
                    
                    m = compute_metrics(pred, depth)
                    sum_mae  += m['mae']  * n
                    sum_rmse += m['rmse'] * n
                    sum_d1   += m['delta1'] * n
            val_loss /= n_val

            # step scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            val_metrics = {'mae': sum_mae / n_val, 'rmse': sum_rmse / n_val, 'delta1': sum_d1 / n_val}
            epochs_val.append(epoch)
            losses_val.append(val_loss)
            metrics_val.append(val_metrics)

            # -- Test --
            test_loss = 0.0
            sum_mae  = 0.0; sum_rmse = 0.0; sum_d1 = 0.0
            with torch.no_grad():
                for event, depth in test_loader:
                    event = event.to(DEVICE); depth = depth.to(DEVICE)
                    pred  = model(event)
                    n = event.size(0)
                    test_loss += combined_loss(pred, depth,
                                 weight_offset=WEIGHT_OFFSET,
                                 berhu_c_frac=BERHU_C_FRAC,
                                 grad_weight=GRAD_WEIGHT).item() * n
                    
                    m = compute_metrics(pred, depth)
                    sum_mae  += m['mae']  * n
                    sum_rmse += m['rmse'] * n
                    sum_d1   += m['delta1'] * n
            test_loss /= n_test
            test_metrics = {'mae': sum_mae / n_test, 'rmse': sum_rmse / n_test, 'delta1': sum_d1 / n_test}
            epochs_test.append(epoch)
            losses_test.append(test_loss)
            metrics_test.append(test_metrics)

            logger.writerow([epoch, f'{train_loss:.6f}',
                             f'{val_loss:.6f}',  f"{val_metrics['mae']:.3f}",  f"{val_metrics['rmse']:.3f}",  f"{val_metrics['delta1']:.3f}",
                             f'{test_loss:.6f}', f"{test_metrics['mae']:.3f}", f"{test_metrics['rmse']:.3f}", f"{test_metrics['delta1']:.3f}", f"{current_lr:.2e}"])
            log_file.flush()

            print(f"Epoch {epoch:04d}/{EPOCHS}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f} (MAE={val_metrics['mae']:.3f}m  RMSE={val_metrics['rmse']:.3f}m  δ<1.25={val_metrics['delta1']:.3f})  "
                  f"test={test_loss:.6f} (MAE={test_metrics['mae']:.3f}m  RMSE={test_metrics['rmse']:.3f}m  δ<1.25={test_metrics['delta1']:.3f})  lr={current_lr:.2e}")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = os.path.join(CKPT_DIR, 'best.pth')
                state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({'epoch': epoch, 'model': state,
                            'val_loss': val_loss, 'metrics': val_metrics}, ckpt_path)
                print(f"    -> saved best checkpoint (val={val_loss:.6f})")

            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} val checks)")
                    break
            # loss + metric curves (updated every val/test check)
            plot_curves(epochs_train, losses_train,
                        epochs_val,   losses_val,  metrics_val,
                        PLOTS_DIR,
                        epochs_test,  losses_test, metrics_test)

            # qualitative grid (every QUAL_EVERY epochs)
            if epoch % QUAL_EVERY == 0 or epoch == 1:
                plot_qualitative(model, test_loader, DEVICE, epoch, PLOTS_DIR,
                                 n_samples=QUAL_SAMPLES, thresh=DEPTH_THRESH,
                                 skip_batches=QUAL_SKIP)

        else:
            print(f'Epoch {epoch}/{EPOCHS}  train={train_loss:.6f}')

    # ── End-of-training plots (reload best checkpoint) ────────────────────
    end = time.time()
    print("Generating final diagnostic plots …")
    best_ckpt_path = os.path.join(CKPT_DIR, 'best.pth')
    best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    plot_model = OrigUNet().to(DEVICE)
    plot_model.load_state_dict(best_ckpt['model'])
    print(f"  (using best.pth — epoch={best_ckpt.get('epoch','?')}, val_loss={best_ckpt.get('val_loss', '?'):.6f})")
    scatter_bands = [
        ('0 - 2.5 m',   0.0,   0.025),
        ('2.5 - 5 m',   0.025, 0.05),
        ('5 - 7.5 m',   0.05,  0.075),
        ('7.5 - 10 m',  0.075, 0.10),
        ('10 - 15 m',   0.1,   0.15),
        ('15 - 20 m',   0.15,  DEPTH_THRESH),
        ('Overall',     0.0,   DEPTH_THRESH),
    ]
    plot_scatter(plot_model, test_loader, DEVICE, PLOTS_DIR, thresh=DEPTH_THRESH, bands=scatter_bands)
    plot_error_histogram(plot_model, test_loader, DEVICE, PLOTS_DIR, thresh=DEPTH_THRESH)

    # Save final checkpoint
    state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({'epoch': EPOCHS, 'model': state,
                'val_loss': losses_val[-1] if losses_val else None},
               os.path.join(CKPT_DIR, 'final.pth'))
    total_secs = int(end - start)
    time_str = f"{total_secs // 3600}h {(total_secs % 3600) // 60}m {total_secs % 60}s"
    logger.writerow(['# Time Taken', time_str, '', '', '', '', '', '', '', '',''])
    log_file.close()

    print(f"Training complete.")
    print(f"Time Taken: {time_str}")
    print(f"Training logs saved to: {log_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == '__main__':
    run()

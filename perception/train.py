import os
import sys
import csv
import time
import optuna

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

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
CKPT_DIR           = os.path.join('/home/srinivasan/ev_nav', 'perception', 'checkpoints')
PLOTS_DIR          = os.path.join('/home/srinivasan/ev_nav', 'perception', 'plots')
EPOCHS        = 200
BATCH_SIZE    = 64          # per GPU
EARLY_STOPPING_PATIENCE = 7
DEPTH_THRESH  = 0.20
WEIGHT_OFFSET = 0.02
WORKERS       = 4
QUAL_EVERY    = 20
QUAL_SAMPLES  = 4
QUAL_SKIP     = 2

# ──────────────────────────────────────────────
# Optuna (load best params if study exists)
# ──────────────────────────────────────────────
try:
    study = optuna.load_study(study_name='unet_depth_cma-es-mod-run1', storage='sqlite:///optuna_study_cma-es-mod-run1.db')
    best = study.best_params
    LR               = best['lr']
    BERHU_C_FRAC     = best['berhu_c_frac']
    GRAD_WEIGHT      = best['grad_weight']
    LSTM_KERNEL_SIZE = best['lstm_kernel_size']
    LSTM_HIDDEN_DIM  = best['lstm_hidden_dim']
    if __name__ == '__main__':
        print(f"Loaded Optuna best params: lr={LR:.2e}, "
              f"berhu_c_frac={BERHU_C_FRAC:.4f}, grad_weight={GRAD_WEIGHT:.4f}, "
              f"lstm_kernel_size={LSTM_KERNEL_SIZE}, lstm_hidden_dim={LSTM_HIDDEN_DIM}")
except Exception:
    LR               = 1e-4
    BERHU_C_FRAC     = 0.2
    GRAD_WEIGHT      = 0.5
    LSTM_KERNEL_SIZE = 1
    LSTM_HIDDEN_DIM  = 512
    if __name__ == '__main__':
        print("No Optuna study found, using default hyperparameters.")


# ──────────────────────────────────────────────
# DDP helpers
# ──────────────────────────────────────────────
def setup_ddp(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


class SequentialDistributedSampler(Sampler):
    """
    Gives each DDP rank a contiguous, non-overlapping chunk of the dataset
    in original (sequential) order.  Required for ConvLSTM temporal training
    so each rank sees a consecutive sequence of frames.

    The last (len(dataset) % world_size) samples are dropped to keep all
    chunks the same length.
    """
    def __init__(self, dataset, num_replicas, rank):
        chunk = len(dataset) // num_replicas
        self.indices = list(range(rank * chunk, (rank + 1) * chunk))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


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
    diff = (pred - target).abs()
    c = berhu_c_frac * diff[mask].max().detach()
    bh = torch.where(diff <= c, diff, (diff**2 + c**2) / (2*c))
    main = (weight * bh).mean()
    grad = gradient_loss(pred, target, thresh)
    return main + grad_weight * grad


def compute_metrics(pred, target, thresh=DEPTH_THRESH):
    mask = (target >= 0) & (target <= thresh)
    if mask.sum() == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'delta1': 0.0}
    p = pred[mask]   * 100.0
    t = target[mask] * 100.0
    mae    = (p - t).abs().mean().item()
    rmse   = ((p - t) ** 2).mean().sqrt().item()
    ratio  = torch.max(p / t.clamp(min=1e-3), t.clamp(min=1e-3) / p.clamp(min=1e-3))
    delta1 = (ratio < 1.25).float().mean().item()
    return {'mae': mae, 'rmse': rmse, 'delta1': delta1}


# ──────────────────────────────────────────────
# Evaluation helper (runs on single device, no DDP)
# ──────────────────────────────────────────────
def evaluate(model, loader, device, n_samples):
    """Returns (loss, metrics_dict) averaged over the full loader.
    Carries ConvLSTM hidden state across batches and resets at episode boundaries,
    matching deployment behaviour."""
    model.eval()
    total_loss = 0.0
    sum_mae = 0.0; sum_rmse = 0.0; sum_d1 = 0.0
    lstm_h       = None
    reset_h_next = True
    with torch.no_grad():
        for event, depth, is_ep_end in loader:
            if reset_h_next:
                lstm_h       = None
                reset_h_next = False
            event = event.to(device); depth = depth.to(device)
            pred, h_new = model(event, lstm_h)
            lstm_h = [[hh.detach(), cc.detach()] for hh, cc in h_new]
            if is_ep_end.any():
                reset_h_next = True
            n = event.size(0)
            total_loss += combined_loss(pred, depth,
                                        weight_offset=WEIGHT_OFFSET,
                                        berhu_c_frac=BERHU_C_FRAC,
                                        grad_weight=GRAD_WEIGHT).item() * n
            m = compute_metrics(pred, depth)
            sum_mae  += m['mae']  * n
            sum_rmse += m['rmse'] * n
            sum_d1   += m['delta1'] * n
    total_loss /= n_samples
    return total_loss, {'mae': sum_mae / n_samples, 'rmse': sum_rmse / n_samples,
                        'delta1': sum_d1 / n_samples}



# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def run(rank, world_size):
    use_ddp = (world_size > 1)
    is_main = (rank == 0)

    if use_ddp:
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ──────────────────────────────────
    # augment=False: random per-frame flips break temporal consistency for LSTM
    # verbose=is_main: suppress duplicate prints from non-zero ranks
    train_dataset = EventDepthDataset(TRAIN_DATASETS_DIR, augment=False, verbose=is_main)
    val_dataset   = EventDepthDataset(VAL_DATASETS_DIR,   augment=False, verbose=is_main)
    test_dataset  = EventDepthDataset(TEST_DATASETS_DIR,  augment=False, verbose=is_main)

    n_train = len(train_dataset)
    n_val   = len(val_dataset)
    n_test  = len(test_dataset)

    if use_ddp:
        # Each rank gets a contiguous sequential chunk — required for ConvLSTM
        train_sampler = SequentialDistributedSampler(train_dataset, world_size, rank)
        train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                   drop_last=True, num_workers=WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  drop_last=True, num_workers=WORKERS, pin_memory=True)

    # Rank-0 keeps full loaders for clean eval and plotting
    if is_main:
        val_loader_full  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=WORKERS, pin_memory=True)
        test_loader_full = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=WORKERS, pin_memory=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        print(f'Train: {n_train}  Val: {n_val}  Test: {n_test}  '
              f'Device: {device}  GPUs: {world_size}  Batch/GPU: {BATCH_SIZE}  '
              f'EffBatch: {BATCH_SIZE * world_size}')

    # ── Model ─────────────────────────────────
    model = OrigUNet(lstm_kernel_size=LSTM_KERNEL_SIZE, lstm_hidden_dim=LSTM_HIDDEN_DIM).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank])

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    patience_counter = 0
    best_val_loss    = float('inf')

    if is_main:
        log_path = os.path.join(PLOTS_DIR, 'train_log.csv')
        log_file = open(log_path, 'w', newline='')
        logger   = csv.writer(log_file)
        logger.writerow(['epoch', 'train_loss',
                         'val_loss',  'val_mae_m',  'val_rmse_m',  'val_delta1',
                         'test_loss', 'test_mae_m', 'test_rmse_m', 'test_delta1', 'lr'])
        epochs_train = []; losses_train = []
        epochs_val   = []; losses_val   = []; metrics_val  = []
        epochs_test  = []; losses_test  = []; metrics_test = []

    # ── Epoch loop ────────────────────────────
    start = time.time()
    for epoch in range(1, EPOCHS + 1):

        # ── Train ─────────────────────────────
        model.train()
        local_loss_sum = torch.tensor(0.0, device=device)
        local_n        = torch.tensor(0,   device=device)
        lstm_h         = None          # reset at epoch start
        reset_h_next   = True          # first batch always starts fresh

        for event, depth, is_ep_end in train_loader:
            if reset_h_next:
                lstm_h       = None
                reset_h_next = False

            event = event.to(device)
            depth = depth.to(device)

            pred, h_new = model(event, lstm_h)
            loss = combined_loss(pred, depth,
                                 weight_offset=WEIGHT_OFFSET,
                                 berhu_c_frac=BERHU_C_FRAC,
                                 grad_weight=GRAD_WEIGHT)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            local_loss_sum += loss.detach() * event.size(0)
            local_n        += event.size(0)

            # TBPTT: detach hidden state — gradients only flow within one batch
            lstm_h = [[hh.detach(), cc.detach()] for hh, cc in h_new]

            # If this batch contained an episode end, start next batch with clean h
            if is_ep_end.any():
                reset_h_next = True

        # Aggregate train loss (all_reduce in DDP, no-op in single-GPU)
        if use_ddp:
            dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_n,        op=dist.ReduceOp.SUM)
        train_loss = (local_loss_sum / local_n).item()

        if is_main:
            epochs_train.append(epoch)
            losses_train.append(train_loss)

        # ── Eval every 5 epochs (rank 0 only, full datasets) ──────────────────
        if epoch % 5 == 0 or epoch == 1:
            val_loss_t  = torch.tensor(float('inf'), device=device)
            stop_signal = torch.tensor(0, device=device)

            if is_main:
                raw_model = model.module if use_ddp else model
                val_loss,  val_metrics  = evaluate(raw_model, val_loader_full,  device, n_val)
                test_loss, test_metrics = evaluate(raw_model, test_loader_full, device, n_test)

                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']

                epochs_val.append(epoch);  losses_val.append(val_loss);   metrics_val.append(val_metrics)
                epochs_test.append(epoch); losses_test.append(test_loss);  metrics_test.append(test_metrics)

                logger.writerow([epoch, f'{train_loss:.6f}',
                                 f'{val_loss:.6f}',  f"{val_metrics['mae']:.3f}",  f"{val_metrics['rmse']:.3f}",  f"{val_metrics['delta1']:.3f}",
                                 f'{test_loss:.6f}', f"{test_metrics['mae']:.3f}", f"{test_metrics['rmse']:.3f}", f"{test_metrics['delta1']:.3f}", f"{current_lr:.2e}"])
                log_file.flush()

                print(f"Epoch {epoch:04d}/{EPOCHS}  train={train_loss:.6f}  "
                      f"val={val_loss:.6f} (MAE={val_metrics['mae']:.3f}m  RMSE={val_metrics['rmse']:.3f}m  δ<1.25={val_metrics['delta1']:.3f})  "
                      f"test={test_loss:.6f} (MAE={test_metrics['mae']:.3f}m  RMSE={test_metrics['rmse']:.3f}m  δ<1.25={test_metrics['delta1']:.3f})  lr={current_lr:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_path = os.path.join(CKPT_DIR, 'best.pth')
                    torch.save({'epoch': epoch, 'model': raw_model.state_dict(),
                                'val_loss': val_loss, 'metrics': val_metrics,
                                'lstm_kernel_size': LSTM_KERNEL_SIZE,
                                'lstm_hidden_dim':  LSTM_HIDDEN_DIM}, ckpt_path)
                    print(f'    -> saved best checkpoint (val={val_loss:.6f})')
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f'Early stopping at epoch {epoch} '
                              f'(no improvement for {EARLY_STOPPING_PATIENCE} val checks)')
                        stop_signal.fill_(1)

                plot_curves(epochs_train, losses_train,
                            epochs_val,  losses_val,  metrics_val,
                            PLOTS_DIR,
                            epochs_test, losses_test, metrics_test)

                if epoch % QUAL_EVERY == 0 or epoch == 1:
                    plot_qualitative(raw_model, test_loader_full, device, epoch, PLOTS_DIR,
                                     n_samples=QUAL_SAMPLES, thresh=DEPTH_THRESH,
                                     skip_batches=QUAL_SKIP)

                val_loss_t.fill_(val_loss)

            if use_ddp:
                # Broadcast val_loss so all ranks step the scheduler identically
                dist.broadcast(val_loss_t,  src=0)
                if not is_main:
                    scheduler.step(val_loss_t.item())
                # Broadcast early-stop signal
                dist.broadcast(stop_signal, src=0)

            if stop_signal.item():
                break

        else:
            if is_main:
                print(f'Epoch {epoch}/{EPOCHS}  train={train_loss:.6f}')

        if use_ddp:
            dist.barrier()

    # ── End-of-training (rank 0 only) ─────────────────────────────────────
    end = time.time()
    if is_main:
        print('Generating final diagnostic plots …')
        best_ckpt = torch.load(os.path.join(CKPT_DIR, 'best.pth'), map_location=device)
        plot_model = OrigUNet(lstm_kernel_size=LSTM_KERNEL_SIZE, lstm_hidden_dim=LSTM_HIDDEN_DIM).to(device)
        plot_model.load_state_dict(best_ckpt['model'])
        print(f"  (using best.pth — epoch={best_ckpt.get('epoch','?')}, "
              f"val_loss={best_ckpt.get('val_loss', '?'):.6f})")

        scatter_bands = [
            ('0 - 2.5 m',   0.0,   0.025),
            ('2.5 - 5 m',   0.025, 0.05),
            ('5 - 7.5 m',   0.05,  0.075),
            ('7.5 - 10 m',  0.075, 0.10),
            ('10 - 15 m',   0.1,   0.15),
            ('15 - 20 m',   0.15,  DEPTH_THRESH),
            ('Overall',     0.0,   DEPTH_THRESH),
        ]
        plot_scatter(plot_model, test_loader_full, device, PLOTS_DIR,
                     thresh=DEPTH_THRESH, bands=scatter_bands)
        plot_error_histogram(plot_model, test_loader_full, device, PLOTS_DIR,
                             thresh=DEPTH_THRESH)

        raw_model = model.module if use_ddp else model
        torch.save({'epoch': epoch, 'model': raw_model.state_dict(),
                    'val_loss': losses_val[-1] if losses_val else None},
                   os.path.join(CKPT_DIR, 'final.pth'))

        total_secs = int(end - start)
        time_str = f"{total_secs // 3600}h {(total_secs % 3600) // 60}m {total_secs % 60}s"
        logger.writerow(['# Time Taken', time_str, '', '', '', '', '', '', '', '', ''])
        log_file.close()

        print(f'Training complete.')
        print(f'Time Taken: {time_str}')
        print(f'Training logs saved to: {os.path.join(PLOTS_DIR, "train_log.csv")}')
        print(f'Plots saved to: {PLOTS_DIR}')

    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    rank       = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    run(rank, world_size)

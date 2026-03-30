import os
import sys
import csv
import time
import torch
import random
import threading
import subprocess
from torch.utils.data import DataLoader, Subset

sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.environ['PROJECT_PATH'])

import optuna
import optuna.visualization as vis

from perception.models import OrigUNet
from perception.dataset import EventDepthDataset
from perception.train import compute_metrics

# CONFIG
TRAIN_DATASETS_DIR = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Train')
VAL_DATASETS_DIR   = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Val')
PLOTS_DIR          = os.path.join('/home/srinivasan/ev_nav', 'perception', 'plots')
EPOCHS        = 100
BATCH_SIZE    = 64
DEPTH_THRESH  = 0.20
WEIGHT_OFFSET = 0.02
WORKERS       = 4
EARLY_STOPPING_PATIENCE = 7
N_TRIALS      = 30

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

# Log
LOG_PATH = os.path.join('/home/srinivasan/ev_nav', 'perception', 'plots', 'tune_log.csv')
log_file = open(LOG_PATH, 'w', newline='')
logger = csv.writer(log_file)
logger.writerow(['trial', 'best_epoch', 'train_loss', 'val_loss', 'val_mae',
                 'lr', 'berhu_c_frac', 'grad_weight', 'lstm_kernel_size', 'lstm_hidden_dim'])

print(f'GPUs available: {n_gpus}  Parallel trials: {n_gpus}  Batch/GPU: {BATCH_SIZE}')
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# Objective
def objective(trial):

    # 1. Assign GPU for this trial — each parallel trial gets its own GPU
    device = torch.device(f'cuda:{trial.number % n_gpus}' if n_gpus > 0 else 'cpu')

    # 2. Sample hyperparameters
    lr               = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    berhu_c_frac     = trial.suggest_float("berhu_c_frac", 0.05, 0.5)
    grad_weight      = trial.suggest_float("grad_weight", 0.0, 2.0)
    lstm_kernel_size = trial.suggest_categorical("lstm_kernel_size", [1, 3])
    lstm_hidden_dim  = trial.suggest_categorical("lstm_hidden_dim", [256, 512])

    # 3. Data — random subset, no temporal ordering needed for HPO
    full_train = EventDepthDataset(TRAIN_DATASETS_DIR, augment=False, verbose=False)
    full_val   = EventDepthDataset(VAL_DATASETS_DIR,   augment=False, verbose=False)
    train_sub  = Subset(full_train, random.sample(range(len(full_train)), 10000))
    val_sub    = Subset(full_val,   random.sample(range(len(full_val)),   3000))
    n_train    = len(train_sub)
    n_val      = len(val_sub)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=WORKERS, pin_memory=True)

    # 4. Model and optimizer — single GPU, no DDP (each trial owns one GPU)
    model     = OrigUNet(lstm_kernel_size=lstm_kernel_size, lstm_hidden_dim=lstm_hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )

    # 5. Local loss using trial hyperparameters
    def _loss(pred, target):
        from torch import where
        mask   = (target >= 0) & (target <= DEPTH_THRESH)
        weight = (1.0 / (target + WEIGHT_OFFSET)) * mask.float()
        diff   = (pred - target).abs()
        c      = berhu_c_frac * diff[mask].max().detach()
        bh     = where(diff <= c, diff, (diff**2 + c**2) / (2*c))
        main   = (weight * bh).mean()
        mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx  = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy  = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad = (pred_dx - tgt_dx).abs()[mask_x].mean() + (pred_dy - tgt_dy).abs()[mask_y].mean()
        return main + grad_weight * grad

    # 6. Training loop
    best_val_mae   = float('inf')
    best_val_loss  = float('inf')
    best_epoch     = 0
    best_train_loss = float('inf')
    patience_counter = 0

    try:
        for epoch in range(1, EPOCHS + 1):

            # Train — no temporal LSTM state for HPO (shuffled subset)
            model.train()
            train_loss = 0.0
            for event, depth, _ in train_loader:
                event, depth = event.to(device), depth.to(device)
                pred, _ = model(event, None)
                loss = _loss(pred, depth)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * event.size(0)
            train_loss /= n_train

            # Validate every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                val_loss = 0.0
                val_mae  = 0.0
                with torch.no_grad():
                    for event, depth, _ in val_loader:
                        event, depth = event.to(device), depth.to(device)
                        pred, _ = model(event, None)
                        n = event.size(0)
                        val_loss += _loss(pred, depth).item() * n
                        val_mae  += compute_metrics(pred, depth)['mae'] * n
                val_loss /= n_val
                val_mae  /= n_val
                scheduler.step(val_loss)

                # Early stopping on val MAE
                if val_mae < best_val_mae:
                    best_val_mae   = val_mae
                    best_val_loss  = val_loss
                    best_epoch     = epoch
                    best_train_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        break

                tprint(f"    Trial {trial.number}  |  Epoch {epoch}/{EPOCHS}  |  "
                       f"train={train_loss:.6f}  val_loss={val_loss:.6f}  val_mae={val_mae:.3f}m")

            else:
                tprint(f"    Trial {trial.number}  |  Epoch {epoch}/{EPOCHS}  |  "
                       f"train={train_loss:.6f}")

    finally:
        logger.writerow([trial.number, best_epoch,
                         f'{best_train_loss:.6f}', f'{best_val_loss:.6f}',
                         f'{best_val_mae:.3f}',
                         f'{lr:.2e}', f'{berhu_c_frac:.4f}', f'{grad_weight:.4f}',
                         lstm_kernel_size, lstm_hidden_dim])
        log_file.flush()

        tprint(f"\n{'='*60}\n"
               f"Trial {trial.number} complete\n"
               f"  Best Epoch:    {best_epoch}\n"
               f"  Best val_mae:  {best_val_mae:.3f}m\n"
               f"  Best val_loss: {best_val_loss:.6f}\n"
               f"  Best train:    {best_train_loss:.6f}\n"
               f"  Params: lr={lr:.2e}, berhu_c_frac={berhu_c_frac:.4f}, "
               f"grad_weight={grad_weight:.4f}, lstm_kernel_size={lstm_kernel_size}, "
               f"lstm_hidden_dim={lstm_hidden_dim}\n"
               f"{'='*60}")

    return best_val_mae


# Run Study
if __name__ == '__main__':
    study = optuna.create_study(
        study_name="unet_depth_cma-es-mod-run1",
        storage="sqlite:///optuna_study_cma-es-mod-run1.db",
        direction="minimize",
        sampler=optuna.samplers.CmaEsSampler(n_startup_trials=4),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    start = time.time()
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=n_gpus)
    total_secs = int(time.time() - start)
    time_str = f"{total_secs // 3600}h {(total_secs % 3600) // 60}m {total_secs % 60}s"

    log_file.close()

    print("\n" + "="*60)
    print(f"Time Taken: {time_str}")
    print(f"Best val_mae: {study.best_value:.3f}m")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}:  {v}")
    print("="*60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Optimization history
    fig = vis.plot_optimization_history(study)
    fig.write_image(os.path.join(PLOTS_DIR, 'optuna_history.png'))

    # Parameter importances
    fig = vis.plot_param_importances(study)
    fig.write_image(os.path.join(PLOTS_DIR, 'optuna_importances.png'))

    # Slice Plot
    fig = vis.plot_slice(study)
    fig.write_image(os.path.join(PLOTS_DIR, 'optuna_slice.png'))

    # Contour
    fig = vis.plot_contour(study)
    fig.write_image(os.path.join(PLOTS_DIR, 'optuna_contour.png'))

    # Parallel coordinate
    fig = vis.plot_parallel_coordinate(study)
    fig.write_image(os.path.join(PLOTS_DIR, 'optuna_parallel.png'))

    # Save best params to file
    with open(os.path.join(PLOTS_DIR, 'best_parameters.txt'), 'w') as f:
        f.write(f"Best val_mae: {study.best_value:.3f}m\n")
        f.write("Best params:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}:  {v}\n")

    # Start full training with best params
    print("\nStarting full training run with best params...")
    env = os.environ.copy()
    train_gpus = 4
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(train_gpus))
    subprocess.run([
        'torchrun', f'--nproc_per_node={train_gpus}',
        '-m', 'perception.train'
    ], env=env)

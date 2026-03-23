import os
import sys
import csv
import time
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data import Subset

sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.environ['PROJECT_PATH'])

import optuna
import optuna.visualization as vis

from perception.models import OrigUNet
from perception.dataset import EventDepthDataset
from perception.train import combined_loss, compute_metrics

# CONFIG
TRAIN_DATASETS_DIR = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Train')
VAL_DATASETS_DIR   = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Val')
PLOTS_DIR          = os.path.join('/home/srinivasan/ev_nav_run2', 'perception', 'plots')
EPOCHS       = 200
BATCH_SIZE   = 64
DEPTH_THRESH = 0.20
WEIGHT_OFFSET = 0.02
WORKERS      = 4
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
EARLY_STOPPING_PATIENCE = 7
N_TRIALS     = 30

# Data
train_dataset = EventDepthDataset(TRAIN_DATASETS_DIR)
indices = random.sample(range(len(train_dataset)), 10000)
train_dataset = Subset(train_dataset, indices)
val_dataset = EventDepthDataset(VAL_DATASETS_DIR)
indices = random.sample(range(len(val_dataset)), 3000)
val_dataset= Subset(val_dataset, indices)
n_train = len(train_dataset)
n_val = len(val_dataset)

# Log
LOG_PATH = os.path.join('/home/srinivasan/ev_nav_run2', 'perception', 'plots', 'tune_log.csv')
log_file = open(LOG_PATH, 'w', newline='')
logger = csv.writer(log_file)
logger.writerow(['trial', 'best_epoch', 'train_loss', 'val_loss', 'val_mae',
                 'lr', 'berhu_c_frac', 'grad_weight'])

n_gpus = min(torch.cuda.device_count() if torch.cuda.is_available() else 1, 4)
effective_batch = BATCH_SIZE * n_gpus

train_loader = DataLoader(train_dataset, batch_size=effective_batch, shuffle=True,
                          num_workers=WORKERS, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=effective_batch, shuffle=True,
                          num_workers=WORKERS, pin_memory=True)

print(f'Train:  {n_train}   Val:  {n_val}     Device:  {DEVICE}   GPUs:  {n_gpus}  Batch:  {effective_batch}')

# Objective
def objective(trial):
    
    # 1. Sample hyperparameters
    lr             = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    berhu_c_frac   = trial.suggest_float("berhu_c_frac", 0.05, 0.5)
    grad_weight    = trial.suggest_float("grad_weight", 0.0, 2.0)

    # 2. Build model and optimizer
    model = OrigUNet().to(DEVICE)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )

    # 3. Training loop
    best_val_mae = float('inf')
    best_val_loss = float('inf')
    best_epoch = 0
    best_train_loss = float('inf')
    patience_counter = 0

    try:
        for epoch in range(1, EPOCHS + 1):

            # Train
            model.train()
            train_loss = 0.0
            for event, depth in train_loader:
                event, depth = event.to(DEVICE), depth.to(DEVICE)
                pred = model(event)
                loss = combined_loss(pred, depth,
                                     weight_offset=WEIGHT_OFFSET,
                                     berhu_c_frac=berhu_c_frac,
                                     grad_weight=grad_weight)
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
                    for event, depth in val_loader:
                        event, depth = event.to(DEVICE), depth.to(DEVICE)
                        pred = model(event)
                        n = event.size(0)
                        val_loss += combined_loss(pred, depth,
                                                  weight_offset=WEIGHT_OFFSET,
                                                  berhu_c_frac=berhu_c_frac,
                                                  grad_weight=grad_weight).item() * n
                        val_mae += compute_metrics(pred, depth)['mae'] * n
                val_loss /= n_val
                val_mae  /= n_val
                scheduler.step(val_loss)

                # Early stopping on val MAE
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_train_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        break

                print(f"    Trial  {trial.number}  |  Epoch  {epoch}/{EPOCHS}  |  "
                  f"train = {train_loss:.6f}  |  val_loss = {val_loss:.6f}  |  val_mae = {val_mae:.3f}m")

            else:
                print(f"    Trial  {trial.number}  |  Epoch  {epoch}/{EPOCHS}  |  "
                    f"train = {train_loss:.6f}")

    finally:
        # Logs even if pruned
        logger.writerow([trial.number, best_epoch,
                         f'{best_train_loss:.6f}', f'{best_val_loss:.6f}',
                         f'{best_val_mae:.3f}',
                         f'{lr:.2e}', f'{berhu_c_frac:.4f}', f'{grad_weight:.4f}'])
        log_file.flush()

        # Print trial summary
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} complete")
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Best val_mae: {best_val_mae:.3f}m")
        print(f"  Best val_loss: {best_val_loss:.6f}")
        print(f"  Best Train_loss : {best_train_loss:.6f}")
        print(f"  Params: lr={lr:.2e}, "
            f"berhu_c_frac={berhu_c_frac:.4f}, grad_weight={grad_weight:.4f}")
        print(f"{'='*60}\n")

    return best_val_mae

# Run Study
if __name__ == '__main__':
    study = optuna.create_study(
        study_name="unet_depth_optuna2",
        storage="sqlite:///optuna_study_run2.db",
        direction="minimize",
        load_if_exists=True
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    start = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
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

    # Save best params to file
    with open(os.path.join(PLOTS_DIR, 'best_parameters.txt'), 'w') as f:
        f.write(f"Best val_mae: {study.best_value:.3f}m\n")
        f.write(f"Best params:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}:  {v}\n")

    # Start full training with best params
    print("\nStarting full training run with best params...")
    import subprocess
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    subprocess.run([sys.executable, '-m', 'perception.train'], env=env)

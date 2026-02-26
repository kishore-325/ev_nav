import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.environ['PROJECT_PATH'])

from perception.models import OrigUNet
from perception.dataset import EventDepthDataset

TEST_DATASETS_DIR = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Test')
CKPT_PATH         = os.path.join(os.environ['PROJECT_PATH'], 'perception', 'checkpoints', 'best.pth')
DEPTH_THRESH      = 0.20
BATCH_SIZE        = 64
WORKERS           = 4
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'


# Depth bands in normalised linear units (value × 100 = metres)
BANDS = [
    ('0 - 5 m',   0.0,  0.05),
    ('5 - 10 m',  0.05, 0.10),
    ('10 - 20 m', 0.10, 0.20),
    ('Overall',   0.0,  DEPTH_THRESH),
]



def evaluate():
    # ── Data ──────────────────────────────────────
    test_dataset = EventDepthDataset(TEST_DATASETS_DIR)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=WORKERS, pin_memory=True)

    # ── Model ─────────────────────────────────────
    model = OrigUNet().to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()

    saved_epoch    = ckpt.get('epoch', '?')
    saved_val_loss = ckpt.get('val_loss', '?')
    val_loss_str = f'{saved_val_loss:.6f}' if isinstance(saved_val_loss, float) else str(saved_val_loss)
    print(f'\nLoaded checkpoint  (epoch={saved_epoch}, val_loss={val_loss_str})')
    print(f'Test samples: {len(test_dataset)}   Device: {DEVICE}\n')

    # ── Accumulate per-pixel predictions ──────────
    # Accumulators: sum of mae/rmse/delta1 contributions and total pixel count per band
    acc = {label: {'sum_mae': 0.0, 'sum_sq': 0.0, 'sum_d1': 0.0, 'n': 0}
           for label, *_ in BANDS}

    with torch.no_grad():
        for event, depth in test_loader:
            event = event.to(DEVICE)
            depth = depth.to(DEVICE)
            pred  = model(event)

            for label, lo, hi in BANDS:
                mask = (depth >= lo) & (depth < hi)
                n = mask.sum().item()
                if n == 0:
                    continue

                p = pred[mask]  * 100.0
                t = depth[mask] * 100.0

                acc[label]['sum_mae'] += (p - t).abs().sum().item()
                acc[label]['sum_sq']  += ((p - t) ** 2).sum().item()
                acc[label]['sum_d1']  += (torch.max(
                    p / t.clamp(min=1e-3),
                    t.clamp(min=1e-3) / p.clamp(min=1e-3)) < 1.25).float().sum().item()
                acc[label]['n']       += n

    # ── Print results ─────────────────────────────
    print(f"{'Depth Band':<14} {'Pixels':>10}  {'MAE (m)':>9}  {'RMSE (m)':>9}  {'delta1':>8}")
    print('─' * 58)
    for label, *_ in BANDS:
        a = acc[label]
        n = a['n']
        if n == 0:
            print(f"{label:<14} {'0':>10}  {'N/A':>9}  {'N/A':>9}  {'N/A':>8}")
            continue
        mae    = a['sum_mae'] / n
        rmse   = (a['sum_sq']  / n) ** 0.5
        delta1 = a['sum_d1']  / n
        sep = '═' * 58 if label == 'Overall' else ''
        if sep:
            print(sep)
        print(f"{label:<14} {n:>10,}  {mae:>9.3f}  {rmse:>9.3f}  {delta1:>8.3f}")

    print()


if __name__ == '__main__':
    evaluate()

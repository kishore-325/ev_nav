import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.environ['PROJECT_PATH'])

from perception.models import MobileNetV3UNet
from perception.dataset import EventDepthDataset

TEST_DATASETS_DIR = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Test')
CKPT_PATH         = os.path.join('/home/srinivasan/ev_nav', 'perception', 'checkpoints', 'best.pth')
DEPTH_THRESH      = 0.20
BATCH_SIZE        = 64
WORKERS           = 4
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'


# Depth bands in normalised linear units (value × 100 = metres)
BANDS = [
    ('0 - 2.5 m',   0.0,  0.025),
    ('2.5 - 5 m',  0.025, 0.05),
    ('5 - 7.5 m', 0.05, 0.075),
    ('7.5 m - 10 m', 0.075, 0.10),
    ('10 - 15 m', 0.1, 0.15),
    ('15 - 20 m', 0.15, 0.20),
    ('Overall',   0.0,  DEPTH_THRESH),
]



def evaluate():
    # ── Data ──────────────────────────────────────
    test_dataset = EventDepthDataset(TEST_DATASETS_DIR)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=WORKERS, pin_memory=True)

    # ── Model ─────────────────────────────────────
    model = MobileNetV3UNet().to(DEVICE)
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

    lstm_h       = None
    reset_h_next = True
    with torch.no_grad():
        for event, depth, is_ep_end in test_loader:
            if reset_h_next:
                lstm_h       = None
                reset_h_next = False
            event = event.to(DEVICE)
            depth = depth.to(DEVICE)
            pred, h_new = model(event, lstm_h)
            lstm_h = [[hh.detach(), cc.detach()] for hh, cc in h_new]
            if is_ep_end.any():
                reset_h_next = True

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
    L, P, M, R, D = 14, 12, 9, 9, 8  # column content widths

    def _sep(l='├', m='┼', r='┤', f='─'):
        return l + f*(L+2) + m + f*(P+2) + m + f*(M+2) + m + f*(R+2) + m + f*(D+2) + r

    def _row(lbl, pix, mae, rmse, d1):
        return f'│ {lbl:<{L}} │ {pix:>{P}} │ {mae:>{M}} │ {rmse:>{R}} │ {d1:>{D}} │'

    print()
    print(_sep('┌', '┬', '┐'))
    print(_row('Depth Band', 'Pixels', 'MAE (m)', 'RMSE (m)', 'delta1'))
    print(_sep())
    for label, *_ in BANDS:
        a = acc[label]
        n = a['n']
        if label == 'Overall':
            print(_sep())
        if n == 0:
            print(_row(label, '0', 'N/A', 'N/A', 'N/A'))
            continue
        mae    = a['sum_mae'] / n
        rmse   = (a['sum_sq']  / n) ** 0.5
        delta1 = a['sum_d1']  / n
        print(_row(label, f'{n:,}', f'{mae:.3f}', f'{rmse:.3f}', f'{delta1:.3f}'))
    print(_sep('└', '┴', '┘'))
    print()


if __name__ == '__main__':
    evaluate()

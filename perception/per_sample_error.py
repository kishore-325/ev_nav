"""
Per-sample depth-band error report.

For every sample in the Test split, run the best model and compute
MAE / RMSE / delta1 / pixel-count within each of the 6 depth bands
(plus an overall column). Writes one CSV row per sample.

Differences from evaluate.py (intentional):
  * loads the .npy pairs directly WITHOUT augmentation (EventDepthDataset
    randomly flips + drops events, which would make per-sample errors
    non-reproducible);
  * uses the local checkpoint at perception/checkpoints/best.pth.

All metrics are in metres (normalised value x 100), matching evaluate.py,
so per-sample numbers aggregate back to the evaluate.txt table.
"""

import os
import sys
import glob
import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Repo root: PROJECT_PATH if set, else the parent of this file's directory.
PROJECT_PATH = os.environ.get('PROJECT_PATH') or \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)
if os.environ.get('FLIGHTMARE_PATH'):
    sys.path.append(os.environ['FLIGHTMARE_PATH'])

from perception.models import OrigUNet

TEST_DIR   = os.path.join(PROJECT_PATH, 'datasets', 'Test')
CKPT_PATH  = os.path.join(PROJECT_PATH, 'perception', 'checkpoints', 'best.pth')
OUT_CSV    = os.path.join(PROJECT_PATH, 'perception', 'plots', 'per_sample_error.csv')
BATCH_SIZE = 64
WORKERS    = 4
DELTA1_THR = 1.25
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# (short_name, label, lo, hi) in normalised units (x100 = metres)
BANDS = [
    ('b1', '0 - 2.5 m',    0.00, 0.025),
    ('b2', '2.5 - 5 m',    0.025, 0.05),
    ('b3', '5 - 7.5 m',    0.05, 0.075),
    ('b4', '7.5 - 10 m',   0.075, 0.10),
    ('b5', '10 - 15 m',    0.10, 0.15),
    ('b6', '15 - 20 m',    0.15, 0.20),
    ('all', 'Overall',     0.00, 0.20),
]


class EvalDataset(Dataset):
    """Paired (event, depth) loader — NO augmentation. Returns (event, depth, index)."""

    def __init__(self, dataset_dir):
        self.samples = []  # (event_path, depth_path, sample_id)
        event_root = os.path.join(dataset_dir, 'EVENTS_RAW')
        depth_root = os.path.join(dataset_dir, 'DEPTH_RAW')

        for env_dir in sorted(os.listdir(event_root)):
            event_env = os.path.join(event_root, env_dir)
            depth_env = os.path.join(depth_root, env_dir)
            if not (os.path.isdir(event_env) and os.path.isdir(depth_env)):
                continue
            for ev_path in sorted(glob.glob(os.path.join(event_env, 'event_raw_sample*.npy'))):
                fname = os.path.basename(ev_path)
                sid = fname.replace('event_raw_sample', '').replace('.npy', '')
                dep_path = os.path.join(depth_env, f'depth_raw_sample{sid}.npy')
                if os.path.exists(dep_path):
                    self.samples.append((ev_path, dep_path, f'{env_dir}/{sid}'))

        if not self.samples:
            raise RuntimeError(f'No paired samples found under {dataset_dir}')
        print(f'[EvalDataset] {len(self.samples)} paired samples found')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ev_path, dep_path, _ = self.samples[index]
        event = torch.from_numpy(np.load(ev_path).astype(np.float32)).unsqueeze(0)  # (1,H,W)
        depth = torch.from_numpy(np.load(dep_path).astype(np.float32)).unsqueeze(0)  # (1,H,W)
        return event, depth, index


def main():
    dataset = EvalDataset(TEST_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=WORKERS, pin_memory=True)

    model = OrigUNet().to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded {CKPT_PATH} (epoch={ckpt.get("epoch", "?")}, '
          f'val_loss={ckpt.get("val_loss", "?")})')
    print(f'Test samples: {len(dataset)}   Device: {DEVICE}')

    # rows[index] = {sample_id, per-band metrics}
    rows = [None] * len(dataset)

    with torch.no_grad():
        for event, depth, idx in loader:
            event = event.to(DEVICE)
            depth = depth.to(DEVICE)
            pred  = model(event)

            p = pred  * 100.0   # (N,1,H,W) metres
            t = depth * 100.0
            ae = (p - t).abs()
            se = (p - t) ** 2
            ratio = torch.max(p / t.clamp(min=1e-3), t / p.clamp(min=1e-3))
            within = (ratio < DELTA1_THR).float()

            for i in range(event.shape[0]):
                sample_i = idx[i].item()
                row = {'sample_id': dataset.samples[sample_i][2]}
                for short, _label, lo, hi in BANDS:
                    m = (depth[i] >= lo) & (depth[i] < hi)
                    n = int(m.sum().item())
                    if n == 0:
                        row[f'{short}_mae']  = ''
                        row[f'{short}_rmse'] = ''
                        row[f'{short}_d1']   = ''
                        row[f'{short}_px']   = 0
                        continue
                    row[f'{short}_mae']  = round(ae[i][m].mean().item(), 6)
                    row[f'{short}_rmse'] = round(se[i][m].mean().item() ** 0.5, 6)
                    row[f'{short}_d1']   = round(within[i][m].mean().item(), 6)
                    row[f'{short}_px']   = n
                rows[sample_i] = row

            done = idx.max().item() + 1
            print(f'\r  processed {done}/{len(dataset)}', end='', flush=True)
    print()

    fieldnames = ['sample_id']
    for short, *_ in BANDS:
        fieldnames += [f'{short}_mae', f'{short}_rmse', f'{short}_d1', f'{short}_px']

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows -> {OUT_CSV}')

    # Sanity check: pixel-weighted aggregate should match evaluate.txt.
    print('\nPixel-weighted aggregate (should match evaluate.txt):')
    for short, label, *_ in BANDS:
        tot_px = sum(r[f'{short}_px'] for r in rows)
        if tot_px == 0:
            continue
        mae  = sum(r[f'{short}_mae']  * r[f'{short}_px'] for r in rows if r[f'{short}_px']) / tot_px
        rmse = (sum(r[f'{short}_rmse']**2 * r[f'{short}_px'] for r in rows if r[f'{short}_px']) / tot_px) ** 0.5
        d1   = sum(r[f'{short}_d1']   * r[f'{short}_px'] for r in rows if r[f'{short}_px']) / tot_px
        print(f'  {label:<14} px={tot_px:>13,}  MAE={mae:.3f}  RMSE={rmse:.3f}  d1={d1:.3f}')


if __name__ == '__main__':
    main()

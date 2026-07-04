"""
Per-sample depth-band ERROR SUMS (numerators, NOT averaged).

One CSV row per Test-split sample, 28 columns = 7 band-groups x 4:
  6 depth bands (0-2.5, 2.5-5, 5-7.5, 7.5-10, 10-15, 15-20 m) + overall (0-20 m).

For each band, only the pixels whose TRUE depth falls in that band are used.
The four columns per band are RAW SUMS (no division by pixel count):

  {b}_sae  = sum of |pred - true|           over band pixels  (-> MAE  = sae/px)
  {b}_sse  = sum of (pred - true)^2         over band pixels  (-> RMSE = sqrt(sse/px))
  {b}_d1n  = count of pixels with max(p/t, t/p) < 1.25         (-> d1   = d1n/px)
  {b}_px   = pixel count in band

The overall (0-20 m) group is computed DIRECTLY over all pixels in that range,
not by summing the six band values.

No augmentation (direct .npy load) -> deterministic. Errors in metres (value x100).
Rows are in the same sorted order as per_sample_error.csv (join by row index).
"""

import os
import sys
import glob
import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_PATH = os.environ.get('PROJECT_PATH') or \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)
if os.environ.get('FLIGHTMARE_PATH'):
    sys.path.append(os.environ['FLIGHTMARE_PATH'])

from perception.models import OrigUNet

TEST_DIR   = os.path.join(PROJECT_PATH, 'datasets', 'Test')
CKPT_PATH  = os.path.join(PROJECT_PATH, 'perception', 'checkpoints', 'best.pth')
OUT_CSV    = os.path.join(PROJECT_PATH, 'perception', 'plots', 'per_sample_band_sums.csv')
BATCH_SIZE = 64
WORKERS    = 4
DELTA1_THR = 1.25
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# (short_name, lo, hi) in normalised units (x100 = metres)
BANDS = [
    ('b1',  0.00, 0.025),
    ('b2',  0.025, 0.05),
    ('b3',  0.05, 0.075),
    ('b4',  0.075, 0.10),
    ('b5',  0.10, 0.15),
    ('b6',  0.15, 0.20),
    ('all', 0.00, 0.20),
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
        event = torch.from_numpy(np.load(ev_path).astype(np.float32)).unsqueeze(0)
        depth = torch.from_numpy(np.load(dep_path).astype(np.float32)).unsqueeze(0)
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
            ratio  = torch.max(p / t.clamp(min=1e-3), t / p.clamp(min=1e-3))
            within = (ratio < DELTA1_THR)

            for i in range(event.shape[0]):
                sample_i = idx[i].item()
                row = {}
                for short, lo, hi in BANDS:
                    m = (depth[i] >= lo) & (depth[i] < hi)   # true-depth mask for this band
                    row[f'{short}_sae'] = round(ae[i][m].sum().item(), 6)      # sum |err|
                    row[f'{short}_sse'] = round(se[i][m].sum().item(), 6)      # sum err^2
                    row[f'{short}_d1n'] = int(within[i][m].sum().item())       # count within thr
                    row[f'{short}_px']  = int(m.sum().item())                  # pixel count
                rows[sample_i] = row

            done = idx.max().item() + 1
            print(f'\r  processed {done}/{len(dataset)}', end='', flush=True)
    print()

    fieldnames = []
    for short, *_ in BANDS:
        fieldnames += [f'{short}_sae', f'{short}_sse', f'{short}_d1n', f'{short}_px']
    assert len(fieldnames) == 28, len(fieldnames)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows x {len(fieldnames)} cols -> {OUT_CSV}')

    # Sanity: aggregate sums -> metrics, should match evaluate.txt table.
    print('\nAggregate from column sums (should match evaluate.txt):')
    for short, *_ in BANDS:
        px = sum(r[f'{short}_px'] for r in rows)
        if px == 0:
            continue
        sae = sum(r[f'{short}_sae'] for r in rows)
        sse = sum(r[f'{short}_sse'] for r in rows)
        d1n = sum(r[f'{short}_d1n'] for r in rows)
        print(f'  {short:<3}  px={px:>13,}  MAE={sae/px:.3f}  '
              f'RMSE={(sse/px)**0.5:.3f}  d1={d1n/px:.3f}')


if __name__ == '__main__':
    main()

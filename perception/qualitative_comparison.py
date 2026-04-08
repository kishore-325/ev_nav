"""
Qualitative comparison plot for paper.

Columns : Event Input (log-diff) | Ground Truth | Baseline | Manual | TPE | CMA-ES
Rows    : sample00017, sample00059, sample00160, sample00260

Checkpoints expected in perception/checkpoints/:
  best_baseline.pth  — Baseline
  best_manual.pth    — Manual (linear-0.20-diverse35-run1)
  best_tpe.pth       — TPE (optuna-run3)   NOTE: user named file best_tpe.pth
  best.pth           — CMA-ES (cma-es-run2)

Run:
    conda activate evfly
    python3 -m perception.qualitative_comparison
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.titlesize':    9,
    'axes.titleweight': 'bold',
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'pdf.fonttype':     42,   # embed fonts as TrueType (IEEE/ACM requirement)
    'ps.fonttype':      42,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.environ['PROJECT_PATH'])

from perception.models import OrigUNet
from perception.model_baseline import OrigUNet as BaselineUNet

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_PATH  = os.environ['PROJECT_PATH']
TEST_DIR      = os.path.join(PROJECT_PATH, 'datasets', 'Test')
CKPT_DIR      = os.path.join(PROJECT_PATH, 'perception', 'checkpoints')
OUT_PATH      = os.path.join(PROJECT_PATH, 'perception', 'plots', 'qualitative_comparison.pdf')

SAMPLE_IDS    = [17, 59, 160, 260]   # sample00017, sample00059, sample00160, sample00260
DEPTH_THRESH  = 0.20                 # normalised; ×100 = metres
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# (display_name, checkpoint_file, arch_key, needs_sigmoid, disp_thresh)
# disp_thresh controls colormap vmax AND whether GT mask is applied:
#   baseline → no GT mask, vmax=0.99 (trained on full depth range)
#   others   → GT mask at 0.20,  vmax=0.20
MODELS = [
    ('Baseline', 'best_baseline.pth', 'baseline', False, 0.99),
    ('Manual',   'best_manual.pth',   'origunet', False, 0.20),
    ('TPE',      'best_tpe.pth',      'origunet', False, 0.20),  # file on disk: best_tpe.pth
    ('CMA-ES',   'best.pth',          'origunet', False, 0.20),
]

COL_TITLES = ['Event Input\n(Log-Difference)', 'Ground Truth',
              'Baseline', 'Manual', 'TPE', 'CMA-ES']


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_sample(sample_idx):
    """Load a single (event, depth) pair directly from .npy — no augmentation."""
    sid = f'{sample_idx:05d}'
    # Dataset iterates sorted(env_dirs); Test has one env directory
    env_dirs = sorted(os.listdir(os.path.join(TEST_DIR, 'EVENTS_RAW')))
    # Find which env contains this index by cumulative count
    offset = 0
    for env in env_dirs:
        ev_env  = os.path.join(TEST_DIR, 'EVENTS_RAW', env)
        dep_env = os.path.join(TEST_DIR, 'DEPTH_RAW',  env)
        ev_path  = os.path.join(ev_env,  f'event_raw_sample{sid}.npy')
        dep_path = os.path.join(dep_env, f'depth_raw_sample{sid}.npy')
        if os.path.exists(ev_path) and os.path.exists(dep_path):
            event = np.load(ev_path).astype(np.float32)   # (260, 346)
            depth = np.load(dep_path).astype(np.float32)  # (260, 346)
            return event, depth
    raise FileNotFoundError(
        f'sample{sid} not found in any environment under {TEST_DIR}')


def load_model(ckpt_name, arch='origunet'):
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    model = (BaselineUNet() if arch == 'baseline' else OrigUNet()).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get('model', ckpt)          # handle plain state-dict saves too
    # Strip DataParallel 'module.' prefix if present
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    print(f'  Loaded {ckpt_name}  arch={arch}  (epoch={epoch})')
    return model


def event_to_rgb(ev):
    """Convert single-channel log-diff event frame to H×W×3 uint8 RGB.
    Positive events → red channel, negative → blue, background → black.
    """
    scale = np.percentile(np.abs(ev), 80)
    ev_sc = np.clip(ev / scale, -1.0, 1.0) if scale > 0 else ev.copy()
    rgb   = np.zeros((*ev.shape, 3), dtype=np.uint8)
    pos   = ev_sc > 0
    neg   = ev_sc < 0
    rgb[pos, 0] = (255 * ev_sc[pos]).astype(np.uint8)
    rgb[neg, 2] = (255 * -ev_sc[neg]).astype(np.uint8)
    return rgb


def run_inference(model, event_np, sigmoid=False):
    """event_np : (H, W) float32 → returns (H, W) float32 pred in [0,1].
    Handles models that return a tuple (pred, hidden_state) (e.g. baseline with ConvLSTM).
    sigmoid=True : apply sigmoid before returning (baseline has no sigmoid in forward).
    """
    tensor = torch.from_numpy(event_np).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
    with torch.no_grad():
        out = model(tensor)
    pred = out[0] if isinstance(out, tuple) else out   # unwrap (pred, h) if needed
    if sigmoid:
        pred = torch.sigmoid(pred)
    return pred.squeeze().cpu().numpy()   # (H, W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Load all models
    print('Loading models ...')
    models = [(name, load_model(ckpt, arch), sig, dt) for name, ckpt, arch, sig, dt in MODELS]

    # Load samples
    print('Loading samples ...')
    samples = []
    for idx in SAMPLE_IDS:
        ev, dep = load_sample(idx)
        samples.append((idx, ev, dep))
        print(f'  sample{idx:05d}  event={ev.shape}  depth={dep.shape}')

    n_rows = len(SAMPLE_IDS)    # 4
    n_cols = len(COL_TITLES)    # 6

    # Image aspect ratio: 260 H × 346 W → each cell height ≈ width × 0.751
    # Target full-width for a two-column IEEE paper (~7.16 in)
    CELL_W  = 7.16 / n_cols          # ≈ 1.19 in per cell
    CELL_H  = CELL_W * (260 / 346)   # ≈ 0.90 in per cell
    FIG_W   = 7.16
    FIG_H   = CELL_H * n_rows + 0.55  # +0.55 for column header row

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Two GridSpecs side by side — narrow gap between col-1 and col-2
    # to visually separate "inputs" (Event, GT) from "predictions"
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.03,
        wspace=0.03,
        left=0.01, right=0.88,
        top=0.93,  bottom=0.01,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    # ── Column headers ────────────────────────────────────────────────────────
    for c, title in enumerate(COL_TITLES):
        axes[0][c].set_title(title, fontsize=8, fontweight='bold',
                             pad=3, linespacing=1.3)

    im_20m = None
    im_99m = None

    for r, (idx, ev_np, dep_np) in enumerate(samples):
        # col 0 — event input
        axes[r][0].imshow(event_to_rgb(ev_np), interpolation='lanczos')

        # col 1 — ground truth (masked to 0–20 m)
        gt_disp = dep_np.copy()
        gt_disp[dep_np >= DEPTH_THRESH] = 0.0
        im_20m = axes[r][1].imshow(gt_disp, cmap='plasma', vmin=0,
                                   vmax=DEPTH_THRESH, interpolation='lanczos')

        # cols 2-5 — model predictions
        for c, (name, model, sig, dt) in enumerate(models, start=2):
            pred      = run_inference(model, ev_np, sigmoid=sig)
            pred_disp = pred.copy()
            if dt < 0.99:
                pred_disp[dep_np >= DEPTH_THRESH] = 0.0
                im = axes[r][c].imshow(pred_disp, cmap='plasma', vmin=0,
                                       vmax=dt, interpolation='lanczos')
                im_20m = im
            else:
                im = axes[r][c].imshow(pred_disp, cmap='plasma', vmin=0,
                                       vmax=dt, interpolation='lanczos')
                im_99m = im

        for c in range(n_cols):
            axes[r][c].axis('off')

    # ── Colorbars ─────────────────────────────────────────────────────────────
    def _add_cbar(im, x, label, vmax, n_ticks=5):
        if im is None:
            return
        cax = fig.add_axes([x, gs.get_subplot_params().bottom,
                            0.012, gs.get_subplot_params().top - gs.get_subplot_params().bottom])
        cb  = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=7, labelpad=3)
        ticks = np.linspace(0, vmax, n_ticks)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{t*100:.0f} m' for t in ticks], fontsize=6)
        cb.ax.tick_params(length=2, pad=1)

    _add_cbar(im_20m, 0.895, 'Depth (GT / Predictions)', DEPTH_THRESH)
    _add_cbar(im_99m, 0.950, 'Depth — Baseline',         0.99)

    fig.savefig(OUT_PATH, bbox_inches='tight', metadata={'Creator': ''})
    plt.close(fig)
    print(f'\nSaved → {OUT_PATH}')


if __name__ == '__main__':
    main()

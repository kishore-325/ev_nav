import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works without a display
import matplotlib.pyplot as plt


def plot_curves(epochs_train, losses_train, epochs_val, losses_val, metrics_val, out_dir):
    """
    PLOT 1 — Loss & metric curves over training.

    Three side-by-side panels:
      Left  : Masked MSE (train + val)  — shows convergence / overfitting gap
      Centre: MAE and RMSE in metres    — interpretable error magnitude
      Right : δ < 1.25 accuracy         — standard depth-estimation benchmark
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- Loss ---
    axes[0].plot(epochs_train, losses_train, label='Train', color='tab:blue', linewidth=1.2)
    if epochs_val:
        axes[0].plot(epochs_val, losses_val, label='Val', color='tab:orange',
                     marker='o', markersize=3, linewidth=1.2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Masked MSE')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- MAE / RMSE ---
    if metrics_val:
        axes[1].plot(epochs_val, [m['mae']  for m in metrics_val],
                     label='MAE (m)',  color='tab:green', marker='o', markersize=3, linewidth=1.2)
        axes[1].plot(epochs_val, [m['rmse'] for m in metrics_val],
                     label='RMSE (m)', color='tab:red',   marker='o', markersize=3, linewidth=1.2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Error (metres)')
        axes[1].set_title('Depth Error Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # --- δ < 1.25 ---
        axes[2].plot(epochs_val, [m['delta1'] for m in metrics_val],
                     color='tab:purple', marker='o', markersize=3, linewidth=1.2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Fraction of valid pixels')
        axes[2].set_title('δ < 1.25 Accuracy\n(fraction where max(p/g, g/p) < 1.25)')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'curves.png'), dpi=120)
    plt.close(fig)


def plot_qualitative(model, val_loader, device, epoch, out_dir, n_samples=4, thresh=0.99):
    """
    PLOT 2 — Side-by-side qualitative grid.

    Columns per row: Event input | GT depth | Predicted depth | |Error| map

    Why it matters: MSE can be low yet the model might be blurring everything.
    Visual inspection catches that immediately.

    Colour maps:
      Event  : RdBu_r  (red = positive events, blue = negative)
      Depth  : plasma  (bright = far, dark = near) — same scale for GT & pred
      Error  : hot     (white = large error)
    """
    model.eval()
    events, depths = next(iter(val_loader))
    events = events[:n_samples].to(device)
    depths = depths[:n_samples].to(device)

    with torch.no_grad():
        preds = model(events)

    events_np = events.cpu().numpy()  # (N, 1, H, W)
    depths_np  = depths.cpu().numpy()
    preds_np   = preds.cpu().numpy()

    fig, axes = plt.subplots(n_samples, 4, figsize=(18, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for c, title in enumerate(['Event Input', 'GT Depth', 'Pred Depth', '|Error| (norm.)']):
        axes[0, c].set_title(title, fontsize=12, fontweight='bold')

    err_im = None
    for i in range(n_samples):
        ev  = events_np[i, 0]   # (H, W)
        gt  = depths_np[i, 0]
        pr  = preds_np[i, 0]
        err = np.abs(pr - gt)
        err[gt >= thresh] = 0.0   # zero out background

        ev_lim = max(abs(ev.min()), abs(ev.max()), 1e-3)
        axes[i, 0].imshow(ev,  cmap='RdBu_r', vmin=-ev_lim, vmax=ev_lim)
        axes[i, 1].imshow(gt,  cmap='plasma',  vmin=0, vmax=1)
        axes[i, 2].imshow(pr,  cmap='plasma',  vmin=0, vmax=1)
        err_im = axes[i, 3].imshow(err, cmap='hot', vmin=0, vmax=0.2)
        for ax in axes[i]:
            ax.axis('off')

    if err_im is not None:
        fig.colorbar(err_im, ax=axes[:, 3], shrink=0.8, label='|error| (normalised, ×100 = metres)')

    fig.suptitle(f'Epoch {epoch}', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'qual_epoch_{epoch:04d}.png'), dpi=100)
    plt.close(fig)


def plot_scatter(model, val_loader, device, out_dir, thresh=0.99):
    """
    PLOT 3 — Scatter: predicted depth vs ground-truth depth (all valid pixels).

    Perfect predictor → all points on red dashed y=x line.
    Systematic bias:
      Points above y=x → model over-estimates distance
      Points below y=x → model under-estimates distance
    Clustering of errors at specific depth ranges reveals where the model struggles.

    Capped at 50 000 pixels for readability.
    """
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for event, depth in val_loader:
            event, depth = event.to(device), depth.to(device)
            pred = model(event)
            mask = depth < thresh
            all_pred.append(pred[mask].cpu().numpy() * 100.0)
            all_gt.append(depth[mask].cpu().numpy()  * 100.0)

    all_pred = np.concatenate(all_pred)
    all_gt   = np.concatenate(all_gt)

    if len(all_pred) > 50_000:
        idx      = np.random.default_rng(0).choice(len(all_pred), 50_000, replace=False)
        all_pred = all_pred[idx]
        all_gt   = all_gt[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_gt, all_pred, alpha=0.2, s=1, color='steelblue', rasterized=True)
    lim = max(all_gt.max(), all_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='y = x  (perfect)')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('GT Depth (m)')
    ax.set_ylabel('Predicted Depth (m)')
    ax.set_title('Predicted vs Ground-Truth Depth\n(valid pixels, val set)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scatter.png'), dpi=120)
    plt.close(fig)


def plot_error_histogram(model, val_loader, device, out_dir, thresh=0.99):
    """
    PLOT 4 — Histogram of absolute depth errors (metres).

    Shows the full error distribution rather than a single mean.
    Useful questions:
      - Is the distribution unimodal or does it have a heavy tail?
      - Mean >> median  →  a small fraction of pixels have catastrophic errors
      - Narrow histogram → consistent, predictable errors (good)
    """
    model.eval()
    all_errors = []
    with torch.no_grad():
        for event, depth in val_loader:
            event, depth = event.to(device), depth.to(device)
            pred = model(event)
            mask = depth < thresh
            err  = (pred[mask] - depth[mask]).abs().cpu().numpy() * 100.0
            all_errors.append(err)

    all_errors = np.concatenate(all_errors)
    mean_err   = float(np.mean(all_errors))
    median_err = float(np.median(all_errors))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_errors, bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(mean_err,   color='orange', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_err:.2f} m')
    ax.axvline(median_err, color='red',    linestyle='--', linewidth=1.5,
               label=f'Median: {median_err:.2f} m')
    ax.set_xlabel('Absolute Error (m)')
    ax.set_ylabel('Pixel count')
    ax.set_title('Depth Error Distribution (val set)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'error_histogram.png'), dpi=120)
    plt.close(fig)

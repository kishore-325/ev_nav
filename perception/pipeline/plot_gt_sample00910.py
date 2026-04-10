"""
Plot ground-truth depth for sample00910 with the same mask and colormap
as plot_sample00910.py, for fair side-by-side comparison.
Saves: perception/plots/gt_sample00910.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEP_PATH     = 'datasets/Test/DEPTH_RAW/environment_1/depth_raw_sample00910.npy'
OUT_PATH     = os.path.join(os.path.dirname(__file__), 'plots', 'gt_sample00910.png')
DEPTH_THRESH = 0.20

depth = np.load(DEP_PATH).astype(np.float32)   # (260, 346)

gt_disp = depth.copy()
gt_disp[depth >= DEPTH_THRESH] = 0.0

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
ax.imshow(gt_disp, cmap='gray', vmin=0, vmax=DEPTH_THRESH)
ax.axis('off')
fig.tight_layout(pad=0)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=120, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f'Saved: {OUT_PATH}')

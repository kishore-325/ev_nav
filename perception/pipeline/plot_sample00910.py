"""
Plot predicted depth for sample00910 using the best CMA-ES checkpoint (best.pth).
Saves: perception/plots/pred_sample00910.png
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from perception.models import OrigUNet

CKPT       = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best.pth')
EV_PATH    = 'datasets/Test/EVENTS_RAW/environment_1/event_raw_sample00910.npy'
DEP_PATH   = 'datasets/Test/DEPTH_RAW/environment_1/depth_raw_sample00910.npy'
OUT_PATH   = os.path.join(os.path.dirname(__file__), 'plots', 'pred_sample00910.png')
DEPTH_THRESH = 0.20
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── load model ────────────────────────────────────────────────────────────────
model = OrigUNet().to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE)
# unwrap checkpoint dict if needed
state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
# handle DataParallel checkpoints
if any(k.startswith('module.') for k in state):
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
print(f'Loaded checkpoint: {CKPT}')

# ── load sample ───────────────────────────────────────────────────────────────
event = np.load(EV_PATH).astype(np.float32)   # (260, 346)
depth = np.load(DEP_PATH).astype(np.float32)  # (260, 346)

event_t = torch.from_numpy(event).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,260,346)

with torch.no_grad():
    pred_t = model(event_t)  # (1,1,260,346)

pred = pred_t.squeeze().cpu().numpy()  # (260, 346)

# mask background pixels for display
pred_disp = pred.copy(); pred_disp[depth >= DEPTH_THRESH] = 0.0

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
ax.imshow(pred_disp, cmap='gray', vmin=0, vmax=DEPTH_THRESH)
ax.axis('off')
fig.tight_layout(pad=0)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=120, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f'Saved: {OUT_PATH}')

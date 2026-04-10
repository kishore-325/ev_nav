import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

# Load the log-difference event frame
sample = np.load('datasets/Test/EVENTS_RAW/environment_1/event_raw_sample00910.npy')

# Visualize as diverging colormap: blue = negative, white = zero, red = positive
vmax = max(abs(sample.min()), abs(sample.max()))

fig, ax = plt.subplots(1, 1, figsize=(3.46, 2.60), dpi=150)
ax.imshow(sample, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig('logdiff_vis_2.png', bbox_inches='tight', pad_inches=0, dpi=150)
print(f"Saved logdiff_vis_2.png  |  shape: {sample.shape}  |  range: [{sample.min():.2f}, {sample.max():.2f}]")

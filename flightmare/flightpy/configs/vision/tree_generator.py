import numpy as np
import os
import re

rng = np.random.default_rng(42)
out_dir = os.path.dirname(os.path.abspath(__file__))
n_configs = 500
start_idx = 200
end_idx   = 400  # exclusive

# ── 1. Generate static_kr CSV files ──────────────────────────────────────────
for obj_idx in range(start_idx, end_idx):
    lines = []
    angles = rng.uniform(0, 2 * np.pi, n_configs)
    xs = rng.uniform(0, 50, n_configs)
    ys = rng.uniform(-20, 10, n_configs)
    for i in range(n_configs):
        qw = np.cos(angles[i] / 2)
        qz = np.sin(angles[i] / 2)
        lines.append(f"rpg_box01,{xs[i]},{ys[i]},0.0,{qw},0.0,{qz},0.0,0.5,0.5,0.5")
    path = os.path.join(out_dir, f"static_kr_{obj_idx}.csv")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written {path}")

print(f"\nDone — generated static_kr_{start_idx}.csv through static_kr_{end_idx-1}.csv ({n_configs} configs each)")

# ── 2. Update dynamic_obstacles.yaml ─────────────────────────────────────────
yaml_path = os.path.join(out_dir, "dynamic_obstacles.yaml")

with open(yaml_path, 'r') as f:
    content = f.read()

# Update N
content = re.sub(r'N: \d+', f'N: {end_idx}', content, count=1)

# Build new object entries
new_entries = []
for obj_idx in range(start_idx + 1, end_idx + 1):  # Object101 ... Object150
    new_entries.append(
        f"Object{obj_idx}:\n"
        f"  csvtraj: traj_00001\n"
        f"  loop: false\n"
        f"  position:\n"
        f"  - 100.0\n"
        f"  - 100.0\n"
        f"  - 100.0\n"
        f"  prefab: rpg_box01\n"
        f"  rotation:\n"
        f"  - 0.0\n"
        f"  - 0.0\n"
        f"  - 0.0\n"
        f"  - 1.0\n"
        f"  scale:\n"
        f"  - 1.0\n"
        f"  - 1.0\n"
        f"  - 1.0"
    )

content = content.rstrip() + "\n" + "\n".join(new_entries) + "\n"

with open(yaml_path, 'w') as f:
    f.write(content)

print(f"Updated {yaml_path}: N={end_idx}, added Object{start_idx+1} through Object{end_idx}")
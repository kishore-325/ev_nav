import os
import re

out_dir   = os.path.dirname(os.path.abspath(__file__))
start_idx = 200   # inclusive
end_idx   = 400   # exclusive

# ── 1. Delete static_kr CSV files ────────────────────────────────────────────
deleted = 0
for obj_idx in range(start_idx, end_idx):
    path = os.path.join(out_dir, f"static_kr_{obj_idx}.csv")
    if os.path.exists(path):
        os.remove(path)
        deleted += 1
    else:
        print(f"Not found (skipped): static_kr_{obj_idx}.csv")

print(f"Deleted {deleted} CSV files (static_kr_{start_idx}.csv -> static_kr_{end_idx-1}.csv)")

# ── 2. Update dynamic_obstacles.yaml ─────────────────────────────────────────
yaml_path = os.path.join(out_dir, "dynamic_obstacles.yaml")

with open(yaml_path, 'r') as f:
    content = f.read()

# Split content preserving the N: line as the first section,
# then each "\nObject{n}:..." block as its own section.
sections = re.split(r'(?=\nObject\d+:)', content)

obj_nums_to_delete = set(range(start_idx + 1, end_idx + 1))
filtered = []
removed_count = 0
for section in sections:
    m = re.match(r'\nObject(\d+):', section)
    if m and int(m.group(1)) in obj_nums_to_delete:
        removed_count += 1
        continue
    filtered.append(section)

content = ''.join(filtered)

# Update N
n_match = re.search(r'N: (\d+)', content)
if n_match:
    old_n = int(n_match.group(1))
    new_n = start_idx
    content = content.replace(f"N: {old_n}", f"N: {new_n}", 1)
    print(f"Updated N: {old_n} → {new_n}")

with open(yaml_path, 'w') as f:
    f.write(content)

print(f"Removed Object{start_idx+1} through Object{end_idx} from yaml ({removed_count} entries)")
print(f"Done.")

import os
import csv
import glob
import time
import numpy as np


DEPTH_THRESH = 0.20   # same as train.py — pixels >= this are background (20 m)
DEPTH_SCALE  = 100.0  # normalised [0,1] -> metres

# Depth bands matching evaluate.py BANDS (in metres)
BANDS = [
    ('0 - 2.5 m',  0.0,   2.5),
    ('2.5 - 5 m',  2.5,   5.0),
    ('5 - 7.5 m',  5.0,   7.5),
    ('7.5 - 10 m', 7.5,  10.0),
    ('10 - 15 m', 10.0,  15.0),
    ('15 - 20 m', 15.0,  20.0),
]


def analyse_split(split_name, dataset_dir):
    """
    Analyse depth distribution for one split (Train, Val, or Test).
    Saves a summary CSV to <dataset_dir>/logs/analysis_<timestamp>.csv
    Returns the summary dict.
    """
    depth_root = os.path.join(dataset_dir, 'DEPTH_RAW')
    if not os.path.isdir(depth_root):
        print(f"[{split_name}] DEPTH_RAW folder not found at {depth_root}")
        return None

    depth_files = sorted(glob.glob(os.path.join(depth_root, '**', 'depth_raw_sample*.npy'),
                                   recursive=True))
    n_samples = len(depth_files)
    if n_samples == 0:
        print(f"[{split_name}] No depth files found under {depth_root}")
        return None

    print(f"[{split_name}] Analysing {n_samples} samples ...")

    all_valid = []
    for path in depth_files:
        depth = np.load(path).astype(np.float32).flatten()
        valid = depth[(depth >= 0) & (depth < DEPTH_THRESH)] * DEPTH_SCALE   # metres
        all_valid.append(valid)

    all_valid = np.concatenate(all_valid)
    total_pixels = len(all_valid)

    depth_min    = float(all_valid.min())
    depth_max    = float(all_valid.max())
    depth_mean   = float(all_valid.mean())
    depth_median = float(np.median(all_valid))

    # Per-band pixel counts and percentages
    band_pcts = {}
    for label, lo, hi in BANDS:
        count = int(((all_valid >= lo) & (all_valid < hi)).sum())
        band_pcts[label] = (count, count / total_pixels * 100)

    # ── Box-drawing table ──────────────────────────────────────────
    L, P, PC = 12, 14, 8   # column widths: label, pixels, percent

    def _sep(l='├', m='┼', r='┤', f='─'):
        return l + f*(L+2) + m + f*(P+2) + m + f*(PC+2) + r

    def _row(lbl, pix, pct):
        return f'│ {lbl:<{L}} │ {pix:>{P}} │ {pct:>{PC}} │'

    print(f"\n  Split : {split_name}   Samples : {n_samples:,}")
    print(f"  Valid pixels (< {DEPTH_THRESH*100:.0f} m) : {total_pixels:,}")
    print(f"  Depth range  : {depth_min:.1f} m – {depth_max:.1f} m")
    print(f"  Mean / Median: {depth_mean:.2f} m / {depth_median:.2f} m")
    print()
    print(_sep('┌', '┬', '┐'))
    print(_row('Band', 'Pixels', '%'))
    print(_sep())
    for label, lo, hi in BANDS:
        count, pct = band_pcts[label]
        print(_row(label, f'{count:,}', f'{pct:.1f}%'))
    print(_sep('└', '┴', '┘'))
    print()

    summary = {
        'split':          split_name,
        'samples':        n_samples,
        'valid_pixels':   total_pixels,
        'depth_min_m':    depth_min,
        'depth_max_m':    depth_max,
        'depth_mean_m':   depth_mean,
        'depth_median_m': depth_median,
        **{f'pct_{label}': pct for label, (_, pct) in band_pcts.items()},
    }

    # ── Save CSV log ───────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_analysis')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"analysis_{split_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'key', 'value'])
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        rows = [
            ('split',        split_name),
            ('samples',      n_samples),
            ('valid_pixels', total_pixels),
            ('depth_min_m',  f'{depth_min:.2f}'),
            ('depth_max_m',  f'{depth_max:.2f}'),
            ('depth_mean_m', f'{depth_mean:.2f}'),
            ('depth_median_m', f'{depth_median:.2f}'),
        ]
        for label, lo, hi in BANDS:
            count, pct = band_pcts[label]
            rows.append((f'pct_{label}', f'{pct:.1f}%'))
        for key, val in rows:
            writer.writerow([ts, key, val])

    print(f"[{split_name}] Log saved to {log_path}")
    return summary


def main():
    project_path = os.environ['PROJECT_PATH']

    train_dir = os.path.join(project_path, 'datasets', 'Train')
    val_dir   = os.path.join(project_path, 'datasets', 'Val')
    test_dir  = os.path.join(project_path, 'datasets', 'Test')

    summaries = {}
    for name, d in [('Train', train_dir), ('Val', val_dir), ('Test', test_dir)]:
        s = analyse_split(name, d)
        if s:
            summaries[name] = s

    if len(summaries) > 1:
        cols = list(summaries.keys())
        L2, P2 = 12, 10

        def _sep2(l='├', m='┼', r='┤', f='─'):
            return l + f*(L2+2) + (m + f*(P2+2)) * len(cols) + r

        def _row2(lbl, *vals):
            return '│ ' + f'{lbl:<{L2}}' + ' │' + ''.join(f' {v:>{P2}} │' for v in vals)

        lines = [
            '\nComparison across splits:',
            _sep2('┌', '┬', '┐'),
            _row2('Band', *cols),
            _sep2(),
        ]
        for label, lo, hi in BANDS:
            vals = [f"{summaries[c][f'pct_{label}']:.1f}%" for c in cols]
            lines.append(_row2(label, *vals))
        lines.append(_sep2('└', '┴', '┘'))

        for line in lines:
            print(line)

        # Save comparison table to collector/dataset_analysis/
        analysis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        cmp_path = os.path.join(analysis_dir, f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(cmp_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nComparison saved to {cmp_path}")


if __name__ == '__main__':
    main()
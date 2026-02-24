import os
import csv
import glob
import time
import numpy as np


DEPTH_THRESH = 0.99   # same as train.py — pixels >= this are background
DEPTH_SCALE  = 100.0  # normalised [0,1] -> metres


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

    pct_0_15  = float((all_valid < 15).mean()                              * 100)
    pct_15_40 = float(((all_valid >= 15) & (all_valid < 40)).mean()        * 100)
    pct_40p   = float((all_valid >= 40).mean()                             * 100)
    depth_min    = float(all_valid.min())
    depth_max    = float(all_valid.max())
    depth_mean   = float(all_valid.mean())
    depth_median = float(np.median(all_valid))

    summary = {
        'split':         split_name,
        'samples':       n_samples,
        'valid_pixels':  total_pixels,
        'pct_0_15m':     pct_0_15,
        'pct_15_40m':    pct_15_40,
        'pct_40m_plus':  pct_40p,
        'depth_min_m':   depth_min,
        'depth_max_m':   depth_max,
        'depth_mean_m':  depth_mean,
        'depth_median_m':depth_median,
    }

    # --- print to console ---
    print(f"\n{'='*45}")
    print(f"  Dataset split : {split_name}")
    print(f"  Samples       : {n_samples:,}")
    print(f"  Valid pixels  : {total_pixels:,}")
    print(f"  Depth range   : {depth_min:.1f}m – {depth_max:.1f}m")
    print(f"  Mean / Median : {depth_mean:.2f}m / {depth_median:.2f}m")
    print(f"  Distribution  :")
    print(f"    0 – 15 m    : {pct_0_15:.1f}%")
    print(f"    15 – 40 m   : {pct_15_40:.1f}%")
    print(f"    40 m+       : {pct_40p:.1f}%")
    print(f"{'='*45}\n")

    # --- save CSV log ---
    log_dir = os.path.join(dataset_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'elapsed_time_s', 'message'])
        ts      = time.strftime('%Y-%m-%d %H:%M:%S')
        start_t = time.time()
        rows = [
            f"split={split_name}",
            f"samples={n_samples}",
            f"valid_pixels={total_pixels}",
            f"depth_range={depth_min:.2f}m-{depth_max:.2f}m",
            f"mean={depth_mean:.2f}m  median={depth_median:.2f}m",
            f"0-15m={pct_0_15:.1f}%  15-40m={pct_15_40:.1f}%  40m+={pct_40p:.1f}%",
        ]
        for r in rows:
            writer.writerow([ts, f'{time.time()-start_t:.3f}', r])

    print(f"[{split_name}] Log saved to {log_path}")
    return summary


def main():
    project_path = os.environ['PROJECT_PATH']

    train_dir = os.path.join(project_path, 'datasets', 'Train')
    val_dir   = os.path.join(project_path, 'datasets', 'Val')
    test_dir  = os.path.join(project_path, 'datasets', 'Test')

    train_summary = analyse_split('Train', train_dir)
    val_summary   = analyse_split('Val',   val_dir)
    test_summary  = analyse_split('Test',  test_dir)

    summaries = {k: v for k, v in [('Train', train_summary), ('Val', val_summary), ('Test', test_summary)] if v}
    if summaries:
        cols = list(summaries.keys())
        print("Summary comparison:")
        print(f"  {'':12s}  " + "  ".join(f"{c:>10s}" for c in cols))
        print(f"  {'Samples':12s}  " + "  ".join(f"{summaries[c]['samples']:>10,}" for c in cols))
        print(f"  {'0-15m %':12s}  " + "  ".join(f"{summaries[c]['pct_0_15m']:>9.1f}%" for c in cols))
        print(f"  {'15-40m %':12s}  " + "  ".join(f"{summaries[c]['pct_15_40m']:>9.1f}%" for c in cols))
        print(f"  {'40m+ %':12s}  " + "  ".join(f"{summaries[c]['pct_40m_plus']:>9.1f}%" for c in cols))


if __name__ == '__main__':
    main()
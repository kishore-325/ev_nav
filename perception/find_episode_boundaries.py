"""
Parse collection logs to extract exact episode boundaries.

At every crash the collector logs:
    "Crash at step X (N/M samples at Z.Zm, V.Vm/s)"
where N is the cumulative number of samples saved in the CURRENT height segment.

At segment completion:
    "Completed M samples at Z.Zm, V.Vm/s (total so far: T)"
where T is the cumulative global sample count across all segments so far.

Algorithm
---------
Process log files in chronological order (filename encodes timestamp).
Carry a `segment_offset` (global index of first sample in current segment).
  - Each crash at N  →  episode end at global index  segment_offset + N - 1
  - Each "Completed … total so far: T"
        →  episode end for the final (non-crash) episode at T - 1
        →  segment_offset = T  (next segment starts here)

Writes  episode_boundaries.json  to each dataset root:
  {
    "episode_end_global_indices": [46, 93, 142, ...],
    "n_episodes": ...,
    "total_samples": ...
  }

Run once before training:
    python3 -m perception.find_episode_boundaries
"""

import os
import sys
import csv
import json
import re
import glob

sys.path.append(os.environ.get('PROJECT_PATH', '.'))

PROJECT_PATH = os.environ.get('PROJECT_PATH', '/home/srinivasan/ev_nav')

DATASET_DIRS = {
    'Train': os.path.join(PROJECT_PATH, 'datasets', 'Train'),
    'Val':   os.path.join(PROJECT_PATH, 'datasets', 'Val'),
    'Test':  os.path.join(PROJECT_PATH, 'datasets', 'Test'),
}

# Regex patterns
RE_CRASH     = re.compile(r'Crash at step \d+ \((\d+)/\d+ samples')
RE_COMPLETED = re.compile(r'Completed \d+ samples.*total so far: (\d+)')


def parse_logs(log_dir):
    """
    Parse all CSV log files in log_dir (sorted chronologically by filename).
    Returns sorted list of 0-indexed global episode-end indices.
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, 'collection_*.csv')))
    if not log_files:
        return None     # no logs → caller falls back

    episode_ends = []
    segment_offset = 0  # global index of first sample in current height segment

    for log_path in log_files:
        last_T = None   # tracks the final "total so far" seen in this log file

        with open(log_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg = row.get('message', '')

                # ── Crash: true episode end (forest config reloaded) ─────────
                m = RE_CRASH.search(msg)
                if m:
                    N = int(m.group(1))
                    # N is the 0-indexed sample_id saved at crash time
                    # → episode ends AT index N (user-confirmed)
                    episode_ends.append(segment_offset + N)
                    continue

                # ── Height-segment completion ────────────────────────────────
                # The drone does NOT teleport between height segments — it just
                # changes altitude in the same forest.  So this is NOT an
                # episode boundary.  We only update segment_offset here so
                # crash counts in the next segment resolve to correct global
                # indices.  The last Completed in each log file IS a session
                # end (drone/sim resets between log files), so we record it.
                m = RE_COMPLETED.search(msg)
                if m:
                    T = int(m.group(1))
                    segment_offset = T   # next height segment starts here
                    last_T = T
                    continue

        # End of this log file: the final episode ran from the last crash to
        # the end of the last height segment — mark it as an episode end.
        if last_T is not None:
            episode_ends.append(last_T - 1)

    return sorted(set(episode_ends)) if episode_ends else None


def count_samples(dataset_dir):
    """Count total paired samples (mirrors EventDepthDataset.__init__ logic)."""
    import glob as _glob
    event_root = os.path.join(dataset_dir, 'EVENTS_RAW')
    depth_root = os.path.join(dataset_dir, 'DEPTH_RAW')
    total = 0
    for env_dir in sorted(os.listdir(event_root)):
        ev_env  = os.path.join(event_root, env_dir)
        dep_env = os.path.join(depth_root, env_dir)
        if not os.path.isdir(ev_env) or not os.path.isdir(dep_env):
            continue
        for ev_path in _glob.glob(os.path.join(ev_env, 'event_raw_sample*.npy')):
            fname = os.path.basename(ev_path)
            sid   = fname.replace('event_raw_sample', '').replace('.npy', '')
            if os.path.exists(os.path.join(dep_env, f'depth_raw_sample{sid}.npy')):
                total += 1
    return total


def main():
    for split, dataset_dir in DATASET_DIRS.items():
        log_dir = os.path.join(dataset_dir, 'logs')

        if not os.path.isdir(log_dir):
            print(f'[{split}] No logs/ directory — skipping.')
            continue

        print(f'\n[{split}] Parsing logs in {log_dir} …')
        episode_ends = parse_logs(log_dir)

        if episode_ends is None:
            print(f'  No collection_*.csv files found — skipping.')
            continue

        total = count_samples(dataset_dir)

        # Ensure the very last sample is marked as an episode end
        if total > 0 and (total - 1) not in episode_ends:
            episode_ends = sorted(set(episode_ends) | {total - 1})

        n_episodes = len(episode_ends)

        # Episode length statistics
        starts  = [0] + [e + 1 for e in episode_ends[:-1]]
        lengths = [episode_ends[i] - starts[i] + 1 for i in range(n_episodes)]
        import numpy as np
        arr = np.array(lengths)
        print(f'  {total} samples  →  {n_episodes} episodes')
        print(f'  Episode length:  min={arr.min()}  median={int(np.median(arr))}'
              f'  mean={arr.mean():.1f}  max={arr.max()}')

        out = {
            'episode_end_global_indices': episode_ends,
            'n_episodes':    n_episodes,
            'total_samples': total,
        }
        out_path = os.path.join(dataset_dir, 'episode_boundaries.json')
        with open(out_path, 'w') as f:
            json.dump(out, f)
        print(f'  Saved → {out_path}')


if __name__ == '__main__':
    main()

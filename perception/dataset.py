import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class EventDepthDataset(Dataset):
    """
    Loads paired (event_raw, depth_raw) numpy arrays from the datasets directory.

    event_raw_sample*.npy : (260, 346) float32, single-channel quantised log-diff
                            values in (−∞, +∞), zero where below threshold
    depth_raw_sample*.npy : (260, 346) float32, normalised depth in [0, 1]
                            metric depth = value × 100 m

    Returns:
        event          : (1, 260, 346) float32 tensor  — single-channel, model splits internally
        depth          : (1, 260, 346) float32 tensor  — normalised [0, 1]
        is_episode_end : scalar bool tensor — True for the LAST frame of an episode.
                         Reset ConvLSTM hidden state BEFORE the next frame.

    Episode boundaries are loaded from  episode_boundaries.json  (produced by
    perception/find_episode_boundaries.py).  If the file does not exist every
    environment-change boundary is still marked, but within-env resets are not.
    """

    def __init__(self, dataset_dir):
        self.samples = []       # list of (event_path, depth_path)
        self._env_end_indices = []   # indices of last sample per environment

        event_root = os.path.join(dataset_dir, 'EVENTS_RAW')
        depth_root = os.path.join(dataset_dir, 'DEPTH_RAW')

        for env_dir in sorted(os.listdir(event_root)):
            event_env = os.path.join(event_root, env_dir)
            depth_env = os.path.join(depth_root, env_dir)
            if not os.path.isdir(event_env) or not os.path.isdir(depth_env):
                continue

            event_files = sorted(glob.glob(os.path.join(event_env, 'event_raw_sample*.npy')))
            for ev_path in event_files:
                fname = os.path.basename(ev_path)
                sample_id = fname.replace('event_raw_sample', '').replace('.npy', '')
                dep_path = os.path.join(depth_env, f'depth_raw_sample{sample_id}.npy')
                if os.path.exists(dep_path):
                    self.samples.append((ev_path, dep_path))

            # last sample of this environment is always an episode end
            if self.samples:
                self._env_end_indices.append(len(self.samples) - 1)

        if len(self.samples) == 0:
            raise RuntimeError(f"No paired samples found under {dataset_dir}")

        # Build episode_end_set from boundaries JSON if available; fall back to
        # environment-level boundaries so the flag is always meaningful.
        boundaries_path = os.path.join(dataset_dir, 'episode_boundaries.json')
        if os.path.exists(boundaries_path):
            with open(boundaries_path) as f:
                data = json.load(f)
            self.episode_end_set = set(data.get('episode_end_global_indices', []))
            n_ep = data.get('n_episodes', len(self.episode_end_set))
            print(f'[EventDepthDataset] {len(self.samples)} samples  |  '
                  f'{n_ep} episodes loaded from {boundaries_path}')
        else:
            self.episode_end_set = set(self._env_end_indices)
            print(f'[EventDepthDataset] {len(self.samples)} samples  |  '
                  f'no episode_boundaries.json — using environment boundaries only '
                  f'({len(self.episode_end_set)} episodes)')

        # Build ordered list-of-lists for external use (sequential LSTM training)
        self.episodes = self._build_episodes()

    def _build_episodes(self):
        """Return [[idx0, idx1, ...], [idxN, ...], ...] sorted by global index."""
        sorted_ends = sorted(self.episode_end_set)
        episodes = []
        start = 0
        for end in sorted_ends:
            episodes.append(list(range(start, end + 1)))
            start = end + 1
        # catch any trailing samples not covered by the boundaries list
        if start < len(self.samples):
            episodes.append(list(range(start, len(self.samples))))
        return episodes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ev_path, dep_path = self.samples[index]

        event = np.load(ev_path).astype(np.float32)     # (260, 346)
        depth = np.load(dep_path).astype(np.float32)    # (260, 346)

        event = torch.from_numpy(event).unsqueeze(0)    # (1, 260, 346)
        depth = torch.from_numpy(depth).unsqueeze(0)    # (1, 260, 346)
        is_episode_end = torch.tensor(index in self.episode_end_set, dtype=torch.bool)

        return event, depth, is_episode_end
    
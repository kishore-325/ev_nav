import os
import json
import glob
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

    Expects episode_boundaries.json at dataset_dir root with key
    'episode_end_global_indices' — list of global sample indices (0-based) that
    are the last frame of each episode. Used to reset ConvLSTM hidden state.

    Returns:
        event      : (1, 260, 346) float32 tensor  — single-channel, model splits internally
        depth      : (1, 260, 346) float32 tensor  — normalised [0, 1]
        is_ep_end  : scalar bool tensor — True if this sample is the last frame of an episode
    """

    def __init__(self, dataset_dir, augment=True, verbose=True):
        self.samples = []       # list of (event_path, depth_path)
        self.augment = augment

        # Load episode boundary indices (global, 0-based, matching self.samples order)
        boundaries_path = os.path.join(dataset_dir, 'episode_boundaries.json')
        if os.path.exists(boundaries_path):
            with open(boundaries_path) as f:
                boundaries = json.load(f)
            self.ep_end_set = set(boundaries['episode_end_global_indices'])
        else:
            self.ep_end_set = set()
            print(f'[EventDepthDataset] Warning: no episode_boundaries.json found at {dataset_dir}')

        event_root = os.path.join(dataset_dir, 'EVENTS_RAW')
        depth_root = os.path.join(dataset_dir, 'DEPTH_RAW')

        for env_dir in sorted(os.listdir(event_root)):
            event_env = os.path.join(event_root, env_dir)
            depth_env = os.path.join(depth_root, env_dir)
            if not os.path.isdir(event_env) or not os.path.isdir(depth_env):
                continue

            event_files = sorted(glob.glob(os.path.join(event_env, 'event_raw_sample*.npy')))
            for ev_path in event_files:
                # derive matching depth path from event filename
                fname = os.path.basename(ev_path)
                sample_id = fname.replace('event_raw_sample','').replace('.npy','')
                dep_path = os.path.join(depth_env, f'depth_raw_sample{sample_id}.npy')
                if os.path.exists(dep_path):
                    self.samples.append((ev_path, dep_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No paired samples found under {dataset_dir}")

        if verbose:
            print(f'[EventDepthDataset] {len(self.samples)} paired samples found  '
                  f'augment={augment}  episodes={len(self.ep_end_set)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ev_path, dep_path = self.samples[index]

        event = np.load(ev_path).astype(np.float32)     # (260, 346)
        depth = np.load(dep_path).astype(np.float32)    # (260, 346)

        event = torch.from_numpy(event).unsqueeze(0)    # (1, 260, 346)
        depth = torch.from_numpy(depth).unsqueeze(0)    # (1, 260, 346)

        if self.augment:
            # Horizontal flip — disabled in temporal mode to keep consecutive frames consistent
            if torch.rand(1) < 0.5:
                event = torch.flip(event, dims=[-1])
                depth = torch.flip(depth, dims=[-1])

            # Event Dropout
            if torch.rand(1) < 0.3:
                drop_mask = torch.rand_like(event) > 0.1
                event = event * drop_mask

        is_ep_end = torch.tensor(index in self.ep_end_set, dtype=torch.bool)
        return event, depth, is_ep_end

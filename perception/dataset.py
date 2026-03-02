import os
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

    Returns:
        event : (1, 260, 346) float32 tensor  — single-channel, model splits internally
        depth : (1, 260, 346) float32 tensor  — normalised [0, 1]
    """

    def __init__(self, dataset_dir):
        self.samples = []       # list of (event_path, depth_path)

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

        print(f'[EventDepthDataset] {len(self.samples)} paired samples found')

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        ev_path, dep_path = self.samples[index]

        # np.load -> loads .npy files
        event = np.load(ev_path).astype(np.float32)     # (260, 346)
        depth = np.load(dep_path).astype(np.float32)    # (260, 346)

        # torch.from_numpy -> converts numpy array to torch tensor
        # unsqueeze(0) -> adds new dimension at pos 0 (260,346) -> (1,260,346)
        event = torch.from_numpy(event).unsqueeze(0)    # (1, 260, 346)
        depth = torch.from_numpy(depth).unsqueeze(0)    # (1, 260, 346)

        # Horizontal flip
        if torch.rand(1) < 0.5:
            event = torch.flip(event, dims=[-1])
            depth = torch.flip(depth, dims=[-1])

        # Event Dropout
        if torch.rand(1) < 0.3:
            drop_mask = torch.rand_like(event) > 0.1
            event = event * drop_mask

        return event, depth
    
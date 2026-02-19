# EV_NAV: Autonomous Drone Navigation with Event Cameras

An autonomous quadrotor navigation system that combines event camera-based perception with deep learning. Uses the Flightmare simulator to collect training data and trains a UNet model to estimate depth from event camera streams for obstacle avoidance.

---

## Project Overview

```
Data Collection (Laptop + Simulator)  →  Transfer Dataset  →  Train UNet (Server GPU)
         ↓                                                           ↓
  RGB, Depth, Events                                          best.pth checkpoint
  Expert velocity labels                                            ↓
                                                          Autonomous Navigation
```

---

## Requirements

- Ubuntu 20.04
- NVIDIA GPU with CUDA 11.3+ support
- Conda (Miniconda or Anaconda)
- 10GB+ free disk space (for dataset)

---

## Quick Start (Training Only — No Simulator Needed)

Use this if you already have a dataset and just want to train the model.

### 1. Clone the Repository

```bash
git clone https://github.com/kishore-325/ev_nav.git
cd ev_nav
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate evfly
```

### 3. Fix PyTorch for Your GPU

Check your GPU CUDA capability first:

```bash
nvidia-smi
```

**For newer GPUs (RTX 30xx, RTX A-series, sm_86+) — CUDA 11.3:**
```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**For older GPUs (RTX 20xx, sm_70 and below) — CUDA 10.2:**
```bash
# Default install from environment.yml is fine
```

### 4. Set Environment Variables

Add to your `~/.bashrc`:
```bash
export FLIGHTMARE_PATH="${HOME}/ev_nav/flightmare"
export PROJECT_PATH="${HOME}/ev_nav"
export PYTHONPATH="${HOME}/ev_nav/flightmare/flightpy"
```

Then:
```bash
source ~/.bashrc
```

### 5. Add Your Dataset

Transfer the `datasets/` folder to `~/ev_nav/datasets/` (not included in repo due to size).

### 6. Train the Model

```bash
conda activate evfly
python perception/train.py
```

Training runs for 200 epochs. Checkpoint saved to `perception/checkpoints/best.pth`.

---

## Full Setup (With Simulator for Data Collection)

Use this if you want to collect new data using the Flightmare simulator.

### Step 1: System Dependencies

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   build-essential \
   cmake \
   libzmqpp-dev \
   libopencv-dev
```

### Step 2: Clone and Set Up Environment

```bash
git clone https://github.com/kishore-325/ev_nav.git
cd ev_nav
conda env create -f environment.yml
conda activate evfly
```

### Step 3: Set Environment Variables

```bash
echo 'export FLIGHTMARE_PATH="${HOME}/ev_nav/flightmare"' >> ~/.bashrc
echo 'export PROJECT_PATH="${HOME}/ev_nav"' >> ~/.bashrc
echo 'export PYTHONPATH="${HOME}/ev_nav/flightmare/flightpy"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Fix CMake Configuration Files

**Fix pybind11 version** — edit `flightmare/flightlib/cmake/pybind11_download.cmake`:

Change line 8 from:
```cmake
GIT_TAG           master
```
to:
```cmake
GIT_TAG           v2.9.2
```

**Fix Google Test branch** — edit `flightmare/flightlib/cmake/gtest_download.cmake`:

Change line 8 from:
```cmake
GIT_TAG           master
```
to:
```cmake
GIT_TAG           main
```

### Step 5: Build and Install Flightlib

```bash
cd ~/ev_nav/flightmare/flightlib

# Clean any previous build artifacts
rm -rf build/ externals/

# Install flightgym (C++ physics engine Python bindings)
pip install .
```

This will download and compile eigen, pybind11, googletest — takes several minutes.

### Step 6: Fix and Install rpg_baselines

Edit `flightmare/flightpy/flightrl/setup.py`:

Change line 20 from:
```python
packages=['rpg_baselines'],
```
to:
```python
packages=find_packages(),
```

Then install:
```bash
cd ~/ev_nav/flightmare/flightpy/flightrl
pip install .
```

### Step 7: Download Unity Renderer

The Unity renderer provides the visual simulation environment.

```bash
mkdir -p ~/ev_nav/flightmare/flightrender
cd ~/ev_nav/flightmare/flightrender

wget https://github.com/uzh-rpg/flightmare/releases/download/0.0.5/RPG_Flightmare.tar.xz
tar -xvf RPG_Flightmare.tar.xz
chmod +x RPG_Flightmare.x86_64
```

### Step 8: Collect Data

**Terminal 1 — Start Unity Renderer:**
```bash
~/ev_nav/flightmare/flightrender/RPG_Flightmare.x86_64
```

**Terminal 2 — Run Data Collector:**
```bash
conda activate evfly
cd ~/ev_nav
python collector/data_label_collector.py
```

Data is saved to `datasets/` with RGB images, depth maps, event streams and expert velocity labels.

---

## Project Structure

```
ev_nav/
├── collector/                  # Data collection scripts
│   ├── data_collector.py       # Basic collection
│   └── data_label_collector.py # Collection with expert labels
├── expert/                     # Expert control policies
│   ├── expert_policy.py        # Grid-based collision avoidance
│   ├── geometric_controller.py # Velocity to thrust/rates controller
│   └── demo_tester.py          # Policy testing
├── perception/                 # Deep learning perception
│   ├── train.py                # UNet training (200 epochs)
│   ├── models.py               # UNet architecture
│   ├── dataset.py              # PyTorch dataset loader
│   ├── plot.py                 # Visualization and metrics
│   └── checkpoints/            # Trained model weights (gitignored)
├── flightmare/                 # Simulator framework
│   ├── flightlib/              # C++ physics engine
│   ├── flightpy/               # Python bindings and configs
│   └── flightrender/           # Unity renderer (download separately)
├── datasets/                   # Training data (gitignored, transfer manually)
│   ├── RGB_IMAGES/
│   ├── DEPTH_RAW/
│   ├── EVENTS_RAW/
│   ├── EVENTS_VIS/
│   └── LABELS/
└── environment.yml             # Conda environment
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Model | UNet (5-layer encoder-decoder) |
| Input | Event camera log-difference (1 channel) |
| Output | Depth map [0, 1] |
| Loss | Masked MSE (ignores background) |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 200 |
| Batch size | 16 |
| Metrics | MAE, RMSE, δ<1.25 accuracy |

---

## Troubleshooting

### GPU not detected / CUDA capability error
Install PyTorch with the correct CUDA version for your GPU (see Step 3 above).

### `flightgym` not found during conda env create
This is a local package. Remove it from `environment.yml` and install manually via `pip install .` in `flightmare/flightlib/`.

### `rpg-baselines` not found during conda env create
Remove it from `environment.yml` and install manually after fixing `setup.py` (see Step 6).

### pybind11 Python version error
Make sure `GIT_TAG v2.9.2` is set in `pybind11_download.cmake`.

### Google Test clone error
Change `GIT_TAG master` to `GIT_TAG main` in `gtest_download.cmake`.

### `gym` version conflict with stable-baselines3
Use `gym==0.19.0` in `environment.yml` (not `gym==0.11.0`).

---

## Package Versions

| Package | Version |
|---------|---------|
| Python | 3.8 |
| PyTorch | 1.10.2+cu113 |
| stable-baselines3 | 1.4.0 |
| gym | 0.19.0 |
| numpy | 1.22.2 |
| opencv-python | 4.5.5.62 |
| pybind11 | v2.9.2 |

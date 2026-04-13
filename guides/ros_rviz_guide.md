# ROS & RViz Learning Guide
### Goal: Live RViz Visualization of OrigUNet Depth Prediction

This guide takes you from zero ROS knowledge to a live RViz demo where your
drone flies in Flightmare, events are computed on-the-fly, your trained OrigUNet
predicts depth, and everything streams in RViz in real time.

Every exercise builds on the previous one. No detours.

---

## Prerequisites

You already have installed:
- ROS Noetic (`/opt/ros/noetic/`)
- `rospy`, `cv_bridge`, `image_transport`
- Python 3, PyTorch, OpenCV, numpy

Before every terminal session, source ROS:
```bash
source /opt/ros/noetic/setup.bash
```

> Add this line to your `~/.bashrc` if you haven't already, so it runs
> automatically:
> ```bash
> echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
> ```

---

## Exercise 1: roscore and the ROS Master

### What you'll learn
- What `roscore` is and why it must be running
- The ROS Master is just a name-lookup service &mdash; nodes register with it,
  and it tells them how to find each other

### How this applies to the final task
Every ROS program needs `roscore` running first. Your visualization node and
RViz both connect through it.

### Do it

**Terminal 1:**
```bash
source /opt/ros/noetic/setup.bash
roscore
```

You'll see output like:
```
... logging to /home/kishore/.ros/log/...
started roslaunch server http://...
started core service [/rosout]
```

This is now running. Leave it open.

**Terminal 2:**
```bash
source /opt/ros/noetic/setup.bash
rosnode list
```

You should see:
```
/rosout
```

That's the only node running &mdash; `rosout` is a built-in logging node that
roscore starts automatically.

```bash
rostopic list
```

Output:
```
/rosout
/rosout_agg
```

These are built-in logging topics. Right now nothing interesting is being
published. That changes in the next exercise.

**Kill roscore** with `Ctrl+C` when done.

### Summary
- `roscore` = the central hub. Must be running before any node starts.
- `rosnode list` = see what nodes are alive.
- `rostopic list` = see what topics exist.

---

## Exercise 2: Your First Publisher Node (Python)

### What you'll learn
- How to write a ROS node in Python using `rospy`
- How to create a **Publisher** that sends messages on a **topic**
- The publish/subscribe pattern: publishers don't know who's listening

### How this applies to the final task
Your final visualization node will be a publisher. It will publish Image
messages on topics like `/event_frame` and `/depth_pred`. This exercise teaches
you the exact pattern.

### Do it

Create a file `~/ev_nav/guides/ex2_publisher.py`:

```python
#!/usr/bin/env python3
"""Exercise 2: A simple ROS publisher that sends numbers."""

import rospy
from std_msgs.msg import Float32

def main():
    # Initialize this script as a ROS node named "my_publisher"
    rospy.init_node("my_publisher")

    # Create a publisher on topic "/my_number" that sends Float32 messages
    pub = rospy.Publisher("/my_number", Float32, queue_size=10)

    # Loop at 2 Hz (2 times per second)
    rate = rospy.Rate(2)
    count = 0.0

    rospy.loginfo("Publisher started. Publishing to /my_number")

    while not rospy.is_shutdown():
        pub.publish(count)
        rospy.loginfo(f"Published: {count}")
        count += 1.0
        rate.sleep()   # sleep to maintain 2 Hz

if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x ~/ev_nav/guides/ex2_publisher.py
```

**Terminal 1:** Start roscore
```bash
source /opt/ros/noetic/setup.bash
roscore
```

**Terminal 2:** Run the publisher
```bash
source /opt/ros/noetic/setup.bash
python3 ~/ev_nav/guides/ex2_publisher.py
```

You'll see it printing "Published: 0.0", "Published: 1.0", etc.

**Terminal 3:** Inspect what's happening
```bash
source /opt/ros/noetic/setup.bash

# See that your node is registered
rosnode list
# Output: /my_publisher, /rosout

# See the topic exists
rostopic list
# Output: /my_number, /rosout, /rosout_agg

# See live data flowing on the topic
rostopic echo /my_number
# You'll see: data: 42.0, data: 43.0, ... streaming in real time
```

Press `Ctrl+C` to stop each.

### Things to understand
- `rospy.init_node("name")` &mdash; registers your script with roscore
- `rospy.Publisher(topic, msg_type, queue_size)` &mdash; creates a publisher
- `rospy.Rate(hz)` + `rate.sleep()` &mdash; controls loop speed
- `rospy.is_shutdown()` &mdash; becomes True when you press Ctrl+C
- `rostopic echo` &mdash; subscribes to a topic and prints every message (your
  go-to debugging tool)

### Summary
- A **node** is just a Python script that calls `rospy.init_node()`
- A **publisher** sends messages to a named **topic**
- Anyone can listen with `rostopic echo` or by writing a subscriber

---

## Exercise 3: Publishing Images (sensor_msgs/Image + cv_bridge)

### What you'll learn
- How to publish **images** (numpy arrays) as ROS Image messages
- `cv_bridge`: the library that converts between OpenCV/numpy and ROS
- The exact message type (`sensor_msgs/Image`) that RViz expects

### How this applies to the final task
Your final node will publish 4 image streams (event frame, GT depth, predicted
depth, error map). This exercise teaches you exactly how to convert a numpy
array to a ROS Image message and publish it.

### Do it

Create `~/ev_nav/guides/ex3_image_publisher.py`:

```python
#!/usr/bin/env python3
"""Exercise 3: Publish a numpy array as a ROS Image."""

import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node("image_publisher")
    pub = rospy.Publisher("/my_image", Image, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(5)  # 5 Hz

    frame = 0
    rospy.loginfo("Image publisher started. Publishing to /my_image")

    while not rospy.is_shutdown():
        # Create a 260x346 gradient image that shifts over time
        img = np.zeros((260, 346), dtype=np.float32)
        for row in range(260):
            img[row, :] = (row + frame) % 260 / 260.0  # value in [0, 1]

        # Convert numpy array to ROS Image message
        # "32FC1" = 32-bit float, 1 channel (same as depth maps)
        msg = bridge.cv2_to_imgmsg(img, encoding="32FC1")
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera"

        pub.publish(msg)
        frame += 5
        rate.sleep()

if __name__ == "__main__":
    main()
```

**Terminal 1:** roscore (if not already running)
**Terminal 2:** Run the node
```bash
source /opt/ros/noetic/setup.bash
python3 ~/ev_nav/guides/ex3_image_publisher.py
```

**Terminal 3:** Verify the topic
```bash
source /opt/ros/noetic/setup.bash
rostopic list          # should show /my_image
rostopic hz /my_image  # should show ~5 Hz
rostopic info /my_image
# Output: Type: sensor_msgs/Image
#         Publishers: /image_publisher
```

You'll view this in RViz in the next exercise.

### Key concepts
- `CvBridge().cv2_to_imgmsg(array, encoding)` converts numpy to ROS Image
- Common encodings you'll use:
  - `"32FC1"` &mdash; float32 single-channel (depth maps, event frames)
  - `"bgr8"` &mdash; 8-bit 3-channel BGR (RGB images from OpenCV)
  - `"mono8"` &mdash; 8-bit grayscale
- `msg.header.stamp` &mdash; timestamp (RViz uses this for synchronization)
- `msg.header.frame_id` &mdash; coordinate frame name (just set to `"camera"`)

### Summary
- `cv_bridge` is the glue between numpy/OpenCV and ROS
- Any numpy array can become a ROS Image with one function call
- This is the exact pattern your final node will use for every image stream

---

## Exercise 4: Viewing Images in RViz

### What you'll learn
- How to open RViz and add image display panels
- How to subscribe an RViz panel to your image topic
- RViz layout: you can have multiple image panels side by side

### How this applies to the final task
In the final demo, you'll have 4 image panels in RViz showing event input,
GT depth, predicted depth, and error. This exercise teaches you to set that up.

### Do it

Keep the image publisher from Exercise 3 running.

**Terminal 3:** Open RViz
```bash
source /opt/ros/noetic/setup.bash
rviz
```

RViz will open with an empty 3D viewport. Now add an Image panel:

1. In the menu bar: **Panels > Add New Panel > (skip this, use displays instead)**

   Actually, for images you add a **Display**:
   - In the left sidebar ("Displays"), click **Add** (bottom-left button)
   - In the popup, select **By display type > Image**
   - Click **OK**

2. In the left sidebar, expand the new **Image** display
   - Set **Image Topic** to `/my_image`
   - You should see the scrolling gradient appear

3. **Add a second Image display** (repeat step 1-2) with the same topic
   - Now you have two panels showing the same image
   - This is how you'll set up 4 panels for the final task

4. **Arrange the panels**: You can drag the Image panels to dock them
   side-by-side. Right-click on an Image display and experiment with placement.

5. **Save your layout**: File > Save Config As > save to
   `~/ev_nav/guides/ex4_layout.rviz`
   - Next time you can open RViz with: `rviz -d ~/ev_nav/guides/ex4_layout.rviz`

### Troubleshooting
- If the image shows as black: check the encoding. `32FC1` with values in
  [0,1] should display as grayscale automatically in RViz's Image display.
- If "No image" appears: make sure the publisher node is still running and
  `rostopic echo /my_image` shows data.
- The `Normalize Range` checkbox in the Image display properties will
  auto-scale float images to visible range. Enable it.

### Summary
- RViz is just a viewer &mdash; it subscribes to topics and renders them
- **Image display** = subscribe to a `sensor_msgs/Image` topic
- You can have as many Image displays as you want
- Save and reload layouts with `.rviz` config files

---

## Exercise 5: Publishing Multiple Image Topics Simultaneously

### What you'll learn
- How to publish multiple image topics from a single node
- How to apply colormaps (for visualizing depth / error as color)
- Pattern: one node, multiple publishers, one loop

### How this applies to the final task
Your final node publishes 4 topics from one loop. This exercise sets up
that exact multi-publisher pattern and teaches you how to colorize depth
and error maps for clear visualization.

### Do it

Create `~/ev_nav/guides/ex5_multi_image.py`:

```python
#!/usr/bin/env python3
"""Exercise 5: Publish multiple image topics from one node."""

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def colorize_depth(depth, max_val=1.0):
    """Convert a float32 depth map [0, max_val] to a BGR8 colormap image."""
    normalized = np.clip(depth / max_val, 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray_u8, cv2.COLORMAP_MAGMA)
    return colored

def colorize_error(error, max_err=0.1):
    """Convert absolute error map to a BGR8 heatmap (red = high error)."""
    normalized = np.clip(error / max_err, 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)
    return colored

def main():
    rospy.init_node("multi_image_publisher")
    bridge = CvBridge()

    # Create 4 publishers — same pattern as the final task
    pub_event = rospy.Publisher("/event_frame", Image, queue_size=1)
    pub_gt    = rospy.Publisher("/depth_gt",    Image, queue_size=1)
    pub_pred  = rospy.Publisher("/depth_pred",  Image, queue_size=1)
    pub_error = rospy.Publisher("/depth_error", Image, queue_size=1)

    rate = rospy.Rate(5)
    frame = 0

    rospy.loginfo("Multi-image publisher started. Topics: /event_frame, /depth_gt, /depth_pred, /depth_error")

    while not rospy.is_shutdown():
        stamp = rospy.Time.now()

        # --- Simulate event frame (2-channel: neg/pos, visualized as red/blue) ---
        event = np.random.randn(260, 346).astype(np.float32) * 0.3
        event_vis = np.zeros((260, 346, 3), dtype=np.uint8)
        event_vis[event > 0, 2] = (np.clip(event[event > 0], 0, 1) * 255).astype(np.uint8)  # red
        event_vis[event < 0, 0] = (np.clip(-event[event < 0], 0, 1) * 255).astype(np.uint8)  # blue

        msg = bridge.cv2_to_imgmsg(event_vis, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        pub_event.publish(msg)

        # --- Simulate GT depth (smooth gradient) ---
        gt_depth = np.zeros((260, 346), dtype=np.float32)
        for r in range(260):
            gt_depth[r, :] = (r + frame) % 260 / 260.0 * 0.2  # [0, 0.2] range

        gt_colored = colorize_depth(gt_depth, max_val=0.2)
        msg = bridge.cv2_to_imgmsg(gt_colored, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        pub_gt.publish(msg)

        # --- Simulate predicted depth (GT + noise) ---
        pred_depth = gt_depth + np.random.randn(260, 346).astype(np.float32) * 0.01
        pred_depth = np.clip(pred_depth, 0, 0.2)

        pred_colored = colorize_depth(pred_depth, max_val=0.2)
        msg = bridge.cv2_to_imgmsg(pred_colored, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        pub_pred.publish(msg)

        # --- Error map ---
        error = np.abs(gt_depth - pred_depth)
        error_colored = colorize_error(error, max_err=0.05)
        msg = bridge.cv2_to_imgmsg(error_colored, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        pub_error.publish(msg)

        frame += 3
        rate.sleep()

if __name__ == "__main__":
    main()
```

Run it:
```bash
source /opt/ros/noetic/setup.bash
python3 ~/ev_nav/guides/ex5_multi_image.py
```

Open RViz and add **4 Image displays**, one for each topic:
- `/event_frame`
- `/depth_gt`
- `/depth_pred`
- `/depth_error`

Arrange them in a 2x2 grid. Save the layout:
```
File > Save Config As > ~/ev_nav/guides/ex5_layout.rviz
```

You should see 4 panels updating simultaneously with fake data.

### Summary
- One node can publish to many topics &mdash; just create multiple `Publisher` objects
- `cv2.applyColorMap()` turns grayscale depth/error into colored images that
  are much easier to read in RViz
- The 4-topic pattern (`/event_frame`, `/depth_gt`, `/depth_pred`, `/depth_error`)
  is exactly what the final task will use

---

## Exercise 6: Publishing Existing Dataset Samples to RViz

### What you'll learn
- How to load your real `.npy` event and depth data and publish it
- How to run your actual OrigUNet model inside a ROS node
- Bridging your PyTorch inference pipeline with ROS publishing

### How this applies to the final task
This is the "perception half" of the final task. You connect your real model
and real data to ROS. The only thing missing after this is the live sim loop
(which you already know how to write from your collectors).

### Do it

Create `~/ev_nav/guides/ex6_model_ros.py`:

```python
#!/usr/bin/env python3
"""Exercise 6: Publish real model predictions to RViz."""

import os
import sys
import glob
import numpy as np
import cv2
import torch

sys.path.append(os.environ['PROJECT_PATH'])

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from perception.models import OrigUNet

# --- Config ---
CKPT_PATH = os.path.join(os.environ['PROJECT_PATH'], 'perception', 'checkpoints', 'best.pth')
TEST_DIR  = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Test')
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

def colorize_depth(depth, max_val=0.2):
    normalized = np.clip(depth / max_val, 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_MAGMA)

def colorize_error(error, max_err=0.05):
    normalized = np.clip(error / max_err, 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)

def make_event_vis(event_raw):
    """Red/blue event visualization (same as test_collector)."""
    scale = np.percentile(np.abs(event_raw), 80)
    if scale > 0:
        events_sc = np.clip(event_raw / scale, -1.0, 1.0)
    else:
        events_sc = event_raw.copy()
    vis = np.zeros((260, 346, 3), dtype=np.uint8)
    pos = events_sc > 0
    neg = events_sc < 0
    vis[pos, 2] = (255 * events_sc[pos]).astype(np.uint8)
    vis[neg, 0] = (255 * -events_sc[neg]).astype(np.uint8)
    return vis

def main():
    rospy.init_node("model_visualizer")
    bridge = CvBridge()

    pub_event = rospy.Publisher("/event_frame", Image, queue_size=1)
    pub_gt    = rospy.Publisher("/depth_gt",    Image, queue_size=1)
    pub_pred  = rospy.Publisher("/depth_pred",  Image, queue_size=1)
    pub_error = rospy.Publisher("/depth_error", Image, queue_size=1)

    # Load model
    model = OrigUNet().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    rospy.loginfo(f"Model loaded from {CKPT_PATH} (epoch={ckpt.get('epoch','?')})")

    # Gather test samples
    event_dir = os.path.join(TEST_DIR, 'EVENTS_RAW', 'environment_1')
    depth_dir = os.path.join(TEST_DIR, 'DEPTH_RAW', 'environment_1')
    event_files = sorted(glob.glob(os.path.join(event_dir, 'event_raw_sample*.npy')))
    rospy.loginfo(f"Found {len(event_files)} test samples")

    rate = rospy.Rate(5)  # 5 Hz playback

    for ev_path in event_files:
        if rospy.is_shutdown():
            break

        # Load data
        sample_id = os.path.basename(ev_path).replace('event_raw_sample','').replace('.npy','')
        dep_path = os.path.join(depth_dir, f'depth_raw_sample{sample_id}.npy')

        event_raw = np.load(ev_path).astype(np.float32)   # (260, 346)
        depth_gt  = np.load(dep_path).astype(np.float32)  # (260, 346)

        # Run inference
        event_tensor = torch.from_numpy(event_raw).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            depth_pred = model(event_tensor).squeeze().cpu().numpy()  # (260, 346)

        stamp = rospy.Time.now()

        # Publish event visualization
        event_vis = make_event_vis(event_raw)
        msg = bridge.cv2_to_imgmsg(event_vis, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        pub_event.publish(msg)

        # Publish GT depth
        msg = bridge.cv2_to_imgmsg(colorize_depth(depth_gt), encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        pub_gt.publish(msg)

        # Publish predicted depth
        msg = bridge.cv2_to_imgmsg(colorize_depth(depth_pred), encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        pub_pred.publish(msg)

        # Publish error
        error = np.abs(depth_gt - depth_pred)
        msg = bridge.cv2_to_imgmsg(colorize_error(error), encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        pub_error.publish(msg)

        rospy.loginfo(f"Sample {sample_id} | MAE={error.mean()*100:.2f}m")
        rate.sleep()

    rospy.loginfo("All samples published. Done.")

if __name__ == "__main__":
    main()
```

Run it:
```bash
# Terminal 1: roscore
# Terminal 2: python3 ~/ev_nav/guides/ex6_model_ros.py
# Terminal 3: rviz -d ~/ev_nav/guides/ex5_layout.rviz
```

You'll see your **real model predictions** streaming through RViz, frame by
frame, with your test dataset.

### Summary
- Your PyTorch model runs inside a ROS node &mdash; no magic, just import and call
- The publish pattern is identical to Exercise 5, just with real data
- This proves the full perception pipeline works over ROS

---

## Exercise 7: Catkin Workspace & Package — The Full Picture

### What you'll learn
- What a catkin workspace is and why ROS needs it
- How to create one from scratch
- What every file and folder inside it does
- How to create a ROS package inside the workspace
- What `package.xml` and `CMakeLists.txt` are (and which parts matter for Python)
- The build-source-run cycle (`catkin_make` → `source` → `roslaunch`)

### How this applies to the final task
To use `roslaunch` (one command to start your node + RViz together), your
scripts must live inside a **catkin package** inside a **catkin workspace**.
This exercise sets all of that up.

---

### Part 1: What is a catkin workspace?

Think of it like a project folder with a specific structure that ROS
understands. Just like Python needs `setup.py` or `pyproject.toml` to know
how to install a package, ROS needs a **catkin workspace** to know where your
code lives and how to build it.

The structure looks like this:
```
~/catkin_ws/                  <-- the workspace root (you pick the name)
├── src/                      <-- YOUR code goes here
│   ├── CMakeLists.txt        <-- auto-generated symlink (don't touch)
│   └── ev_nav_viz/           <-- your package (a folder with specific files)
│       ├── CMakeLists.txt    <-- build instructions
│       ├── package.xml       <-- package metadata + dependencies
│       ├── scripts/          <-- your Python scripts go here
│       ├── launch/           <-- .launch files go here
│       └── rviz/             <-- saved RViz layouts go here
│
├── build/                    <-- auto-generated by catkin_make (don't touch)
├── devel/                    <-- auto-generated by catkin_make (don't touch)
│   └── setup.bash            <-- YOU SOURCE THIS to activate the workspace
└── logs/                     <-- build logs (don't touch)
```

**Key idea:** You only ever work inside `src/`. The `build/` and `devel/`
folders are auto-generated when you run `catkin_make`. You never edit them.

---

### Part 2: Create the workspace

**Step 1: Make the directory structure**
```bash
mkdir -p ~/catkin_ws/src
```

That's it. Just one folder with a `src/` inside.

**Step 2: Initialize and build it**
```bash
cd ~/catkin_ws
catkin_make
```

This does three things:
- Creates the `build/` folder (where compilation happens)
- Creates the `devel/` folder (where the "installed" result lives)
- Creates `src/CMakeLists.txt` (a symlink ROS needs internally)

You'll see output ending with:
```
[100%] Built target ...
```

**Step 3: Source the workspace**
```bash
source ~/catkin_ws/devel/setup.bash
```

This tells your terminal "hey, ROS packages inside `~/catkin_ws/` exist now."
Without this, ROS can't find your packages.

**IMPORTANT:** You need to run this `source` command in every new terminal.
Add it to your `~/.bashrc` so it happens automatically:

```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

> **Note:** If you already have `source /opt/ros/noetic/setup.bash` in your
> `~/.bashrc`, the catkin workspace source **must come after it**. The catkin
> workspace source automatically includes the ROS base, so technically you
> only need the catkin one. But having both is fine as long as catkin comes
> second.

**Step 4: Verify it works**
```bash
# Open a new terminal (or source ~/.bashrc)
echo $ROS_PACKAGE_PATH
```

You should see something like:
```
/home/kishore/catkin_ws/src:/opt/ros/noetic/share
```

Both your workspace AND the ROS system packages are visible. If you only see
the `/opt/ros/...` part, the source command didn't work.

---

### Part 3: What is a catkin package?

A **package** is a folder inside `src/` that contains two special files:
- `package.xml` &mdash; "who am I and what do I depend on?"
- `CMakeLists.txt` &mdash; "how to build me"

That's the minimum. A folder without these two files is NOT a ROS package and
ROS will ignore it.

For Python-only packages (like yours), these files are mostly boilerplate.
The real code lives in `scripts/`.

---

### Part 4: Create your package

```bash
cd ~/catkin_ws/src
catkin_create_pkg ev_nav_viz rospy sensor_msgs std_msgs cv_bridge
```

Breaking down this command:
- `catkin_create_pkg` &mdash; the tool that creates a package
- `ev_nav_viz` &mdash; your package name (you pick this)
- `rospy sensor_msgs std_msgs cv_bridge` &mdash; dependencies your package needs

This creates:
```
~/catkin_ws/src/ev_nav_viz/
├── CMakeLists.txt
├── package.xml
├── include/
│   └── ev_nav_viz/
└── src/
```

Now add the folders you actually need:
```bash
mkdir -p ~/catkin_ws/src/ev_nav_viz/scripts
mkdir -p ~/catkin_ws/src/ev_nav_viz/launch
mkdir -p ~/catkin_ws/src/ev_nav_viz/rviz
```

Your package now looks like:
```
~/catkin_ws/src/ev_nav_viz/
├── CMakeLists.txt       <-- build instructions (mostly boilerplate for Python)
├── package.xml          <-- metadata + dependencies
├── scripts/             <-- your Python ROS nodes go here
├── launch/              <-- .launch files go here
├── rviz/                <-- saved RViz config files go here
├── include/             <-- (for C++ headers, you won't use this)
└── src/                 <-- (for C++ source files, you won't use this)
```

---

### Part 5: Understanding package.xml

Open `~/catkin_ws/src/ev_nav_viz/package.xml`. It was auto-generated and
looks like this (simplified):

```xml
<?xml version="1.0"?>
<package format="2">
  <name>ev_nav_viz</name>
  <version>0.0.0</version>
  <description>The ev_nav_viz package</description>
  <maintainer email="kishore@todo.todo">kishore</maintainer>
  <license>TODO</license>

  <buildtool_depend>catkin</buildtool_depend>

  <depend>rospy</depend>
  <depend>sensor_msgs</depend>
  <depend>std_msgs</depend>
  <depend>cv_bridge</depend>
</package>
```

What each part means:
- `<name>` &mdash; must match the folder name
- `<depend>` &mdash; the ROS packages yours needs. These are the dependencies you
  listed in `catkin_create_pkg`. They tell ROS "make sure these are
  available before trying to use my package"
- The rest (version, description, license) is metadata. You can ignore it.

**When to edit this file:** If you later need a new ROS dependency (e.g.,
`geometry_msgs`), add a `<depend>geometry_msgs</depend>` line.

---

### Part 6: Understanding CMakeLists.txt (for Python packages)

The auto-generated `CMakeLists.txt` is long and full of comments. For a
**Python-only** package, most of it is irrelevant. The important parts are:

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(ev_nav_viz)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
)

catkin_package()

# This line tells catkin to install your Python scripts
catkin_install_python(PROGRAMS
  scripts/live_demo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

**The key line is `catkin_install_python`**. It tells ROS "these Python scripts
are executable nodes." Without this, `roslaunch` won't find your scripts.

> **You'll need to edit this** after adding scripts. For now, don't worry about
> it &mdash; we'll update it when we add scripts in the next steps.

However, for development/testing, there's a simpler approach: just make your
scripts executable with `chmod +x` and `roslaunch` will find them directly in
the `scripts/` folder. The `catkin_install_python` line is mainly needed for
proper installation. Both approaches work during development.

---

### Part 7: Add a script and test the package

Copy your Exercise 5 script into the package:
```bash
cp ~/ev_nav/guides/ex5_multi_image.py ~/catkin_ws/src/ev_nav_viz/scripts/multi_image.py
chmod +x ~/catkin_ws/src/ev_nav_viz/scripts/multi_image.py
```

The `chmod +x` is **critical**. ROS finds Python scripts by looking for
executable files in the `scripts/` directory. Without it, `roslaunch` will
fail with "cannot find node."

Now rebuild:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

**Why rebuild?** `catkin_make` re-scans `src/` and registers any changes
(new packages, new files, updated CMakeLists). You need to rebuild whenever
you add a new package or change `CMakeLists.txt` or `package.xml`.

> **You do NOT need to rebuild** when you just edit a Python script's content.
> Python is interpreted, so changes take effect immediately. Just restart
> the node.

Verify ROS can find your package:
```bash
rospack find ev_nav_viz
# Should print: /home/kishore/catkin_ws/src/ev_nav_viz
```

If it says "not found", you forgot to `source devel/setup.bash`.

---

### Part 8: Build-Source-Run cycle (the mental model)

Here's the cycle you'll use from now on:

```
  Edit code in ~/catkin_ws/src/ev_nav_viz/scripts/
       │
       ▼
  Did you change package.xml, CMakeLists.txt, or add a new package?
       │                              │
      YES                             NO
       │                              │
       ▼                              ▼
  cd ~/catkin_ws                 Just restart the node
  catkin_make                    (Ctrl+C and re-run roslaunch)
  source devel/setup.bash
       │
       ▼
  roslaunch ev_nav_viz demo.launch
```

**Summary of when to rebuild:**
| What changed                  | Need `catkin_make`? | Need `source`? |
|-------------------------------|---------------------|----------------|
| Edited a Python script        | No                  | No             |
| Added a new Python script     | Yes (update CMake)  | Yes            |
| Changed `package.xml`         | Yes                 | Yes            |
| Changed `CMakeLists.txt`      | Yes                 | Yes            |
| Created a whole new package   | Yes                 | Yes            |

---

### Summary
- **Catkin workspace** = `~/catkin_ws/` with `src/`, `build/`, `devel/`
- You only work in `src/`. The rest is auto-generated.
- **Catkin package** = a folder in `src/` with `package.xml` + `CMakeLists.txt`
- `catkin_make` builds everything. `source devel/setup.bash` activates it.
- For Python: just put scripts in `scripts/`, `chmod +x` them, rebuild once.
- After that, editing Python files doesn't require rebuilding.

---

## Exercise 8: Launch Files (Starting Everything at Once)

### What you'll learn
- How to write a `.launch` file that starts multiple nodes together
- `roslaunch` starts `roscore` automatically (no separate terminal needed)
- Launch file syntax: `<node>`, `<arg>`, `$(find ...)`, `output="screen"`
- How to pass environment variables and arguments

### How this applies to the final task
Instead of opening 3 terminals manually (roscore, your node, rviz), you'll run
one command: `roslaunch ev_nav_viz demo.launch` and everything starts together.

---

### Part 1: What is a launch file?

A `.launch` file is XML that tells ROS "start these nodes together." Think of
it as a script that replaces opening multiple terminals manually.

Without a launch file:
```bash
# Terminal 1:
roscore
# Terminal 2:
python3 my_node.py
# Terminal 3:
rviz
```

With a launch file:
```bash
# Single terminal:
roslaunch ev_nav_viz demo.launch
```

`roslaunch` automatically:
- Starts `roscore` if it's not already running
- Starts all nodes listed in the file
- Kills everything cleanly when you press Ctrl+C

---

### Part 2: Anatomy of a launch file

```xml
<launch>
    <!-- Everything goes inside <launch> tags -->

    <!-- Start a Python node -->
    <node name="my_node"
          pkg="ev_nav_viz"
          type="multi_image.py"
          output="screen" />

    <!-- Start RViz with a saved layout -->
    <node name="rviz"
          pkg="rviz"
          type="rviz"
          args="-d $(find ev_nav_viz)/rviz/demo.rviz" />
</launch>
```

**`<node>` attributes explained:**

| Attribute          | What it means                                         | Example                   |
|--------------------|-------------------------------------------------------|---------------------------|
| `name`             | Name shown in `rosnode list`                          | `"live_depth_demo"`       |
| `pkg`              | Which ROS package this node belongs to                | `"ev_nav_viz"`            |
| `type`             | The executable file name (must be in `scripts/`)      | `"live_demo.py"`          |
| `output="screen"`  | Print node's output to your terminal                  | (vs. hiding it in a log)  |
| `args`             | Command-line arguments passed to the executable       | `"-d path/to/file.rviz"`  |

**Special expressions in launch files:**

| Expression            | What it does                                          | Example output                              |
|-----------------------|-------------------------------------------------------|---------------------------------------------|
| `$(find pkg_name)`    | Returns the path to a ROS package                     | `/home/kishore/catkin_ws/src/ev_nav_viz`     |
| `$(env VAR_NAME)`     | Reads a shell environment variable                    | `$(env HOME)` → `/home/kishore`             |
| `$(arg arg_name)`     | References a launch file argument (see Part 4)        | whatever value was passed                   |

---

### Part 3: Create and test a launch file

Create `~/catkin_ws/src/ev_nav_viz/launch/test.launch`:

```xml
<launch>
    <!-- Start the multi-image publisher from Exercise 5 -->
    <node name="multi_image"
          pkg="ev_nav_viz"
          type="multi_image.py"
          output="screen" />

    <!-- Start RViz -->
    <node name="rviz"
          pkg="rviz"
          type="rviz" />
</launch>
```

Make sure the workspace is built and sourced:
```bash
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

Run it:
```bash
roslaunch ev_nav_viz test.launch
```

You should see:
1. Your node starts and prints "Multi-image publisher started..."
2. RViz opens (add Image displays manually for now)
3. Both are running from **one terminal, one command**

Press `Ctrl+C` to kill everything cleanly.

---

### Part 4: Launch file arguments

Arguments let you customize a launch file without editing it. Useful for
things like "should I open RViz or not?" or "which checkpoint to use?"

```xml
<launch>
    <!-- Declare an argument with a default value -->
    <arg name="open_rviz" default="true" />
    <arg name="publish_hz" default="10" />

    <!-- Use the argument -->
    <node name="multi_image"
          pkg="ev_nav_viz"
          type="multi_image.py"
          output="screen" />

    <!-- Conditionally start RViz based on the argument -->
    <node name="rviz"
          pkg="rviz"
          type="rviz"
          if="$(arg open_rviz)" />
</launch>
```

Override from the command line:
```bash
# With RViz (default)
roslaunch ev_nav_viz test.launch

# Without RViz
roslaunch ev_nav_viz test.launch open_rviz:=false
```

---

### Part 5: Passing environment variables to nodes

Your existing code uses `os.environ['PROJECT_PATH']` and
`os.environ['FLIGHTMARE_PATH']`. Launch files can set these:

```xml
<launch>
    <node name="live_demo"
          pkg="ev_nav_viz"
          type="live_demo.py"
          output="screen">

        <!-- Set environment variables for this node -->
        <env name="PROJECT_PATH" value="/home/kishore/ev_nav" />
        <env name="FLIGHTMARE_PATH" value="/home/kishore/ev_nav/flightmare" />
    </node>
</launch>
```

This is optional &mdash; if these variables are already set in your `~/.bashrc`,
the node will inherit them automatically. But it's good practice to be
explicit in the launch file so it works on any machine.

---

### Part 6: Including other launch files

You can include one launch file inside another:

```xml
<launch>
    <!-- Include another launch file -->
    <include file="$(find ev_nav_viz)/launch/test.launch">
        <arg name="open_rviz" value="false" />
    </include>
</launch>
```

You won't need this right now, but it's how larger ROS systems are organized.

---

### Exercise: verify the full chain works

1. Make sure your workspace is built:
   ```bash
   cd ~/catkin_ws && catkin_make && source devel/setup.bash
   ```

2. Run the test launch:
   ```bash
   roslaunch ev_nav_viz test.launch
   ```

3. In another terminal, verify everything is running:
   ```bash
   source ~/catkin_ws/devel/setup.bash
   rosnode list
   # Should show: /multi_image, /rviz, /rosout

   rostopic list
   # Should show: /event_frame, /depth_gt, /depth_pred, /depth_error, ...

   rostopic hz /event_frame
   # Should show: ~5 Hz
   ```

4. Kill everything with `Ctrl+C` in the launch terminal

If all of this works, you're ready for the final exercise.

### Summary
- **Launch file** = XML that starts multiple nodes with one command
- `roslaunch` auto-starts `roscore` (no separate terminal needed!)
- `<node>` entries specify: name, package, executable, output mode
- `$(find pkg)` locates packages, `$(arg name)` reads arguments
- `<env>` tags set environment variables for nodes
- `Ctrl+C` kills everything cleanly

---

## Exercise 9 (Final): Live Flightmare + OrigUNet + RViz

### What you'll learn
- How to integrate the Flightmare sim loop with ROS publishing
- Running the expert policy, computing events on-the-fly, running inference,
  and streaming results to RViz &mdash; all in one node
- Everything from exercises 1-8 combined

### How this applies to the final task
**This IS the final task.**

### Architecture recap

```
Flightmare Unity
      |
[Your ROS Node]
  1. env.step(expert_action)     <-- from expert_policy.py
  2. env.getImage() -> RGB       <-- grayscale -> log-diff
  3. compute_events()            <-- from your collectors
  4. model(events)               <-- OrigUNet inference
  5. env.getDepthImage()         <-- ground truth
  6. Publish all to ROS topics   <-- exercises 3-5
      |
    RViz  (4 image panels)       <-- exercise 4
```

### Do it

Create `~/catkin_ws/src/ev_nav_viz/scripts/live_demo.py`:

```python
#!/usr/bin/env python3
"""
Live RViz visualization: Flightmare + ExpertPolicy + OrigUNet depth prediction.

Runs the simulator, flies with the expert policy, computes events on-the-fly,
runs OrigUNet inference, and publishes everything to RViz.
"""

import os
import sys
import numpy as np
import cv2
import torch

# --- Path setup (same as your collectors) ---
sys.path.append(os.environ['FLIGHTMARE_PATH'])
sys.path.append(os.path.join(os.environ['FLIGHTMARE_PATH'], 'flightlib'))
sys.path.append(os.path.join(os.environ['FLIGHTMARE_PATH'], 'flightpy', 'flightrl'))
sys.path.append(os.environ['PROJECT_PATH'])

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ruamel.yaml import YAML, dump, RoundTripDumper
from flightgym import VisionEnv_v1
from rpg_baselines.torch.envs.vec_env_wrapper import FlightEnvVec
from expert.expert_policy import ExpertPolicy
from expert.geometric_controller import GeometricVelocityController
from perception.models import OrigUNet

# --- Config ---
CKPT_PATH     = os.path.join(os.environ['PROJECT_PATH'], 'perception', 'checkpoints', 'best.pth')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
DESIRED_VEL   = 5.0
TARGET_HEIGHT = 3.0
PUBLISH_HZ    = 10       # how fast to publish to RViz
SMALL_EPS     = 1e-5
POS_THRESH    = 0.2
NEG_THRESH    = 0.2


def colorize_depth(depth, max_val=0.2):
    """Depth map [0, max_val] -> BGR8 colormap."""
    normalized = np.clip(depth / max_val, 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_MAGMA)


def colorize_error(error, max_err=0.05):
    """Absolute error -> BGR8 heatmap."""
    normalized = np.clip(error / max_err, 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)


def make_event_vis(event_raw):
    """Red/blue event visualization."""
    scale = np.percentile(np.abs(event_raw), 80)
    if scale > 0:
        events_sc = np.clip(event_raw / scale, -1.0, 1.0)
    else:
        events_sc = event_raw.copy()
    vis = np.zeros((260, 346, 3), dtype=np.uint8)
    pos = events_sc > 0
    neg = events_sc < 0
    vis[pos, 2] = (255 * events_sc[pos]).astype(np.uint8)
    vis[neg, 0] = (255 * -events_sc[neg]).astype(np.uint8)
    return vis


def compute_events(prev_gray, gray):
    """Compute quantized log-difference events (same as your collectors)."""
    difflog = np.log(gray + SMALL_EPS) - np.log(prev_gray + SMALL_EPS)
    events = np.zeros_like(difflog)
    if np.abs(difflog).max() >= max(POS_THRESH, NEG_THRESH):
        pos_idx = np.where(difflog > 0.0)
        neg_idx = np.where(difflog < 0.0)
        events[pos_idx] = (difflog[pos_idx] // POS_THRESH) * POS_THRESH
        events[neg_idx] = (difflog[neg_idx] // -NEG_THRESH) * -NEG_THRESH
    return events


def create_env():
    """Create Flightmare environment (same as your collectors)."""
    yaml = YAML()
    config_path = os.path.join(
        os.environ['FLIGHTMARE_PATH'],
        'flightpy/configs/vision/config.yaml'
    )
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    cfg['simulation']['num_envs'] = 1
    cfg_str = dump(cfg, Dumper=RoundTripDumper)
    raw_env = VisionEnv_v1(cfg_str, False)
    return FlightEnvVec(raw_env)


def stabilize_height(env, controller, target_z=3.0, height_tol=0.2,
                     kp=2.0, kd=0.5, max_vz=2.0, max_steps=500, frame_id=0):
    """Fly drone to target height before starting (from your collectors)."""
    for step in range(max_steps):
        env.getObs()
        quad_state = env.getQuadState()
        current_z = quad_state[0][3]
        current_vz = quad_state[0][10]
        height_err = target_z - current_z
        vel_z_cmd = float(np.clip(kp * height_err - kd * current_vz, -max_vz, max_vz))

        if abs(height_err) <= height_tol:
            vel_cmd_world = np.array([0.0, 0.0, vel_z_cmd])
        else:
            vel_cmd_world = np.array([1.0, 0.0, vel_z_cmd])

        action = controller.compute_action(vel_cmd_world, quad_state[0])
        env.step(np.array([action]))
        env.render(frame_id)
        frame_id += 1

        if abs(height_err) <= height_tol:
            rospy.loginfo(f"Height stabilized at z={current_z:.2f}m")
            return True, frame_id

    return False, frame_id


def main():
    rospy.init_node("live_depth_demo")
    bridge = CvBridge()

    # --- Publishers ---
    pub_event = rospy.Publisher("/event_frame", Image, queue_size=1)
    pub_gt    = rospy.Publisher("/depth_gt",    Image, queue_size=1)
    pub_pred  = rospy.Publisher("/depth_pred",  Image, queue_size=1)
    pub_error = rospy.Publisher("/depth_error", Image, queue_size=1)

    # --- Load model ---
    model = OrigUNet().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    rospy.loginfo(f"OrigUNet loaded (epoch={ckpt.get('epoch','?')}, device={DEVICE})")

    # --- Create environment ---
    env = create_env()
    controller = GeometricVelocityController()
    expert = ExpertPolicy(desired_vel=DESIRED_VEL)

    rospy.loginfo("Connecting to Unity...")
    env.connectUnity()
    env.reset()
    env.move()

    frame_id = 0
    env.render(frame_id)
    frame_id += 1

    # Stabilize height
    rospy.loginfo(f"Stabilizing at height {TARGET_HEIGHT}m...")
    stable, frame_id = stabilize_height(env, controller, TARGET_HEIGHT, frame_id=frame_id)
    if not stable:
        rospy.logwarn("Could not stabilize height, proceeding anyway")

    # --- Main loop ---
    rate = rospy.Rate(PUBLISH_HZ)
    ref_gray = None
    capture_interval = 3
    capture_gap = 35
    pair_stride = capture_interval + capture_gap
    step = 0
    sample_count = 0

    rospy.loginfo("Starting live visualization loop...")
    rospy.loginfo("Open RViz and add Image displays for: /event_frame, /depth_gt, /depth_pred, /depth_error")

    while not rospy.is_shutdown():
        # --- Expert policy step (same as your collectors) ---
        env.getObs()
        raw_obs = env._observation
        quad_state = env.getQuadState()

        vel_cmd_body, extras = expert.compute_expert_velocity(raw_obs[0])
        R_wb = raw_obs[0, 3:12].reshape(3, 3)
        vel_cmd_world = R_wb.T @ vel_cmd_body
        vel_cmd_world[2] = 0.0
        action = controller.compute_action(vel_cmd_world, quad_state[0])

        step_mod = step % pair_stride

        # Phase 1: capture reference frame
        if step > 0 and step_mod == capture_interval:
            rgb_flat = env.getImage(rgb=True)
            rgb = rgb_flat.reshape(1, 260, 346, 3)
            ref_gray = cv2.cvtColor(rgb[0], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Phase 2: capture current frame, compute events, run model, publish
        elif step > 0 and step_mod == (2 * capture_interval) % pair_stride and ref_gray is not None:
            rgb_flat = env.getImage(rgb=True)
            rgb = rgb_flat.reshape(1, 260, 346, 3)
            gray = cv2.cvtColor(rgb[0], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            # Compute events (same as your collectors)
            event_raw = compute_events(ref_gray, gray)

            # Get ground truth depth
            depth_flat = env.getDepthImage()
            depth_gt = depth_flat.reshape(260, 346)

            # Run OrigUNet inference
            event_tensor = torch.from_numpy(event_raw).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                depth_pred = model(event_tensor).squeeze().cpu().numpy()

            # --- Publish everything ---
            stamp = rospy.Time.now()

            # Event visualization
            event_vis = make_event_vis(event_raw)
            msg = bridge.cv2_to_imgmsg(event_vis, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = "camera"
            pub_event.publish(msg)

            # GT depth
            msg = bridge.cv2_to_imgmsg(colorize_depth(depth_gt), encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = "camera"
            pub_gt.publish(msg)

            # Predicted depth
            msg = bridge.cv2_to_imgmsg(colorize_depth(depth_pred), encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = "camera"
            pub_pred.publish(msg)

            # Error map
            error = np.abs(depth_gt - depth_pred)
            msg = bridge.cv2_to_imgmsg(colorize_error(error), encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = "camera"
            pub_error.publish(msg)

            sample_count += 1
            rospy.loginfo(f"Sample {sample_count} | MAE={error.mean()*100:.2f}m | "
                          f"obstacles={extras['num_obstacles_detected']}")

        # Step the simulation
        _, _, done, _ = env.step(np.array([action]))
        env.render(frame_id)
        frame_id += 1
        step += 1

        # Handle crash
        if done.any():
            rospy.logwarn("Crash! Resetting environment...")
            env.reset()
            env.move()
            env.render(frame_id)
            frame_id += 1
            ref_gray = None
            stable, frame_id = stabilize_height(env, controller, TARGET_HEIGHT, frame_id=frame_id)

        rate.sleep()

    # Cleanup
    env.disconnectUnity()
    env.close()

if __name__ == "__main__":
    main()
```

Make it executable and set up the package:
```bash
chmod +x ~/catkin_ws/src/ev_nav_viz/scripts/live_demo.py
```

Create the launch file `~/catkin_ws/src/ev_nav_viz/launch/demo.launch`:
```xml
<launch>
    <node name="live_depth_demo" pkg="ev_nav_viz" type="live_demo.py"
          output="screen" />

    <node name="rviz" pkg="rviz" type="rviz"
          args="-d $(find ev_nav_viz)/rviz/demo.rviz" />
</launch>
```

Build and run:
```bash
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

### Running the full demo

**Option A: Using roslaunch (recommended)**
```bash
# Make sure Unity/Flightmare renderer is running first
roslaunch ev_nav_viz demo.launch
```

**Option B: Manual (if you want more control)**
```bash
# Terminal 1: roscore
roscore

# Terminal 2: Start Flightmare Unity renderer

# Terminal 3: Run the node
source ~/catkin_ws/devel/setup.bash
python3 ~/catkin_ws/src/ev_nav_viz/scripts/live_demo.py

# Terminal 4: RViz
rviz
# Add 4 Image displays for the 4 topics
```

### What you'll see in RViz

| Panel          | Topic           | Shows                                        |
|----------------|-----------------|----------------------------------------------|
| Top-left       | /event_frame    | Red/blue event visualization (model input)   |
| Top-right      | /depth_gt       | Ground truth depth from simulator (magma)    |
| Bottom-left    | /depth_pred     | OrigUNet predicted depth (magma)             |
| Bottom-right   | /depth_error    | Absolute error heatmap (jet, red=high error) |

The drone flies through the forest using your expert policy, and you see in
real time how well the model predicts depth from events alone.

### Summary
This final exercise combines:
- **Ex 1-2**: ROS node, roscore, publishers
- **Ex 3**: cv_bridge for image publishing
- **Ex 4**: RViz Image displays
- **Ex 5**: Multiple topics from one node
- **Ex 6**: Real model inference inside a ROS node
- **Ex 7-8**: Launch file and catkin package

---

## Quick Reference

### Commands you'll use most
```bash
roscore                          # start ROS master
rosnode list                     # see running nodes
rostopic list                    # see active topics
rostopic echo /topic_name        # see live data on a topic
rostopic hz /topic_name          # check publish rate
rostopic info /topic_name        # see type + publishers/subscribers
roslaunch package file.launch    # start everything from a launch file
rviz                             # open RViz
rviz -d path/to/layout.rviz     # open RViz with saved layout
```

### Debugging checklist
1. Is `roscore` running? (`rosnode list` should return something)
2. Is your node running? (`rosnode list` should show your node name)
3. Is data flowing? (`rostopic hz /your_topic` should show a rate)
4. Is RViz subscribed? (check the Image display topic dropdown)
5. Is the encoding correct? (float images need `Normalize Range` checked in RViz)
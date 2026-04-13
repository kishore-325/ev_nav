import os
import cv2
import sys
import torch
import numpy as np
import glob

sys.path.append(os.environ['PROJECT_PATH'])

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from perception.models import OrigUNet

CKPT_PATH = os.path.join(os.environ['PROJECT_PATH'], 'perception', 'checkpoints', 'best.pth')
TEST_DIR = os.path.join(os.environ['PROJECT_PATH'], 'datasets', 'Test')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def colorize_error(err, max_err = 0.05):
    normalized = np.clip(err/max_err, 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_HOT)


def compute_eve_vis(eve_raw):
    scale = np.percentile(np.abs(eve_raw), 80)
    if scale > 0:
        eve_raw = np.clip(eve_raw/scale, -1.0, 1.0)
    eve_vis = np.zeros((260, 346, 3), dtype=np.uint8)
    pos = eve_raw > 0
    neg = eve_raw < 0
    eve_vis[pos, 2] = (255 * eve_raw[pos]).astype(np.uint8)
    eve_vis[neg, 0] = (255 * -eve_raw[neg]).astype(np.uint8)
    return eve_vis

def main():

    rospy.init_node("model_visualizer")
    bridge = CvBridge()

    # Data
    eve_dir = os.path.join(TEST_DIR, 'EVENTS_RAW', 'environment_1')
    dep_dir = os.path.join(TEST_DIR, 'DEPTH_RAW', 'environment_1')
    eve_files = sorted(glob.glob(os.path.join(eve_dir, 'event_raw_sample*.npy')))
    rospy.loginfo(f"Found {len(eve_files)} test samples")

    #model
    model = OrigUNet().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    rospy.loginfo(f"Model loaded from {CKPT_PATH} (epoch={ckpt.get('epoch', '?')})")

    #publishers
    eve_pub = rospy.Publisher("/event_frame", Image, queue_size=1)
    gt_pub = rospy.Publisher("/gt_depth", Image, queue_size=1)
    pred_pub = rospy.Publisher("/pred_depth", Image, queue_size=1)
    err_pub = rospy.Publisher("/error_frame", Image, queue_size=1)

    rate = rospy.Rate(3)

    #loop
    for eve_path in eve_files:

        if rospy.is_shutdown():
            break

        sample_id = os.path.basename(eve_path).replace('event_raw_sample', '').replace('.npy', '')
        dep_path = os.path.join(dep_dir, f'depth_raw_sample{sample_id}.npy')

        eve_raw = np.load(eve_path).astype(np.float32)
        dep_raw = np.load(dep_path).astype(np.float32)

        eve_tensor = torch.from_numpy(eve_raw).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dep_pred = model(eve_tensor).squeeze().cpu().numpy()

        stamp = rospy.Time.now()

        #Publish event frame
        eve_vis = compute_eve_vis(eve_raw)
        msg = bridge.cv2_to_imgmsg(eve_vis, encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera'
        eve_pub.publish(msg)

        
        #Publish ground truth depth
        msg = bridge.cv2_to_imgmsg(dep_raw, encoding='32FC1')
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera'
        gt_pub.publish(msg)


        #Publish predicted depth
        msg = bridge.cv2_to_imgmsg(dep_pred, encoding='32FC1')
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera'
        pred_pub.publish(msg)


        #Publish error frame
        err = np.abs(dep_raw - dep_pred)
        msg = bridge.cv2_to_imgmsg(colorize_error(err), encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera'
        err_pub.publish(msg)


        rospy.loginfo(f"Sample {sample_id} | MAE = {err.mean()* 100:.2f}m")
        rate.sleep()

    rospy.loginfo("All samples has been published.")

if __name__ == "__main__":
    main()

    
        





import numpy as np
import rospy
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node("image_publisher")

    depth_pub = rospy.Publisher("/my_depth", Image, queue_size=1)
    event_pub = rospy.Publisher("/my_event", Image, queue_size=1)

    bridge = CvBridge()
    rate = rospy.Rate(20)

    sample = 0
    rospy.loginfo("Image Publisher started. Publishing to /my_depth and /my_event")

    dep_dir = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/DEPTH_RAW/environment_1')
    eve_dir = os.path.join(os.environ['PROJECT_PATH'], 'datasets/Test/EVENTS_RAW/environment_1')

    while not rospy.is_shutdown() and sample <= 10799:

        # Convert numpy array to ROS Image message
        # "32FC1" = 32-bit float, 1 channel (same as depth maps)
        dep_raw = np.load(os.path.join(dep_dir, f"depth_raw_sample{sample:05d}.npy")).astype(np.float32)
        dep_msg = bridge.cv2_to_imgmsg(dep_raw, encoding="32FC1")
        dep_msg.header.stamp = rospy.Time.now()
        dep_msg.header.frame_id = "camera"
        depth_pub.publish(dep_msg)

        eve_raw = np.load(os.path.join(eve_dir, f"event_raw_sample{sample:05d}.npy")).astype(np.float32)
        scale = np.percentile(np.abs(eve_raw), 80)
        if scale > 0:
            eve_raw = np.clip(eve_raw/scale, -1.0, 1.0)
        eve_vis = np.zeros((260, 346, 3), dtype=np.uint8)
        pos = eve_raw > 0
        neg = eve_raw < 0
        eve_vis[pos, 2] = (255 * eve_raw[pos]).astype(np.uint8)
        eve_vis[neg, 0] = (255 * -eve_raw[neg]).astype(np.uint8)
        eve_msg = bridge.cv2_to_imgmsg(eve_vis, encoding="bgr8")
        eve_msg.header.stamp = rospy.Time.now()
        eve_msg.header.frame_id = "camera"
        event_pub.publish(eve_msg)


        sample += 1
        rate.sleep()

if __name__ == "__main__":
    main()

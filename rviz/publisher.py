import rospy
from std_msgs.msg import Float32

def main():

    rospy.init_node("my_publisher")

    pub = rospy.Publisher("/my_number", Float32, queue_size=10)

    rate = rospy.Rate(2)
    count = 0.0

    rospy.loginfo("Publisher started. Publishing to /my_number")

    while not rospy.is_shutdown():
        pub.publish(count)
        rospy.loginfo(f"Published: {count}")
        count += 1.0
        rate.sleep()

if __name__ == "__main__":
    main()


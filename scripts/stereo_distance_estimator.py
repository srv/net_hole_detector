#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def callback(msg):
    for point in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
        x, y, z = point
        print("Distancia:", z)
        break

rospy.init_node("distance_reader")
rospy.Subscriber("/stereo/points2", PointCloud2, callback)
rospy.spin()
#!/usr/bin/env python3
import rospy
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from statistics import mean, median
import numpy as np
from cv_bridge import CvBridge

HYSTERESI = 50

class DisparityRepublisher:

    def __init__(self):
        rospy.Subscriber("/stereo/disparity", DisparityImage, self.callback_disparity)
        rospy.Subscriber("/stereo/points2", PointCloud2, self.callback_distance)
        self.pub_disp = rospy.Publisher("/stereo/disparity_image", Image, queue_size=1)
        self.pub_deb = rospy.Publisher("/stereo/deb_image", Image, queue_size=1)

        self.bridge = CvBridge()


    def callback_disparity(self, msg):
        self.pub_disp.publish(msg.image)

    def callback_distance(self, msg):
        height = msg.height
        width = msg.width
        new_point_cloud = np.zeros((height, width), dtype=np.uint8)

        points = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=False))
        point_cloud = np.array(points)

        i_height = int(height / 2) - int(HYSTERESI)
        i_width = int(width / 2)
        for i in range(HYSTERESI*2):         

            punt_1 = int(height * (width/2 - HYSTERESI + i) + (width/2 - HYSTERESI) - 1)
            punt_2 = int(height * (width/2 - HYSTERESI + i) + (width/2 + HYSTERESI) - 1)

            if i == 0:
                parcial = point_cloud[punt_1:punt_2, 2]
            else:
                parcial = np.append(parcial, point_cloud[punt_1:punt_2, 2])
            new_point_cloud[i_height, i_width-HYSTERESI:i_width+HYSTERESI] = 255
            i_height += 1

        #print(parcial)
        image_msg = Image()
        image_msg = self.bridge.cv2_to_imgmsg(new_point_cloud, encoding="mono8")
        image_msg.header = msg.header
        self.pub_deb.publish(image_msg)

        print("Distancia:" + str(median(parcial[~np.isnan(parcial)])))


if __name__ == "__main__":
    rospy.init_node("distance_reader")
    DisparityRepublisher()
    rospy.spin()
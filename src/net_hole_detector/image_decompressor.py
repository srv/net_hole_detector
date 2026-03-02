#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageDecompressor:
    def __init__(self):
        rospy.init_node("image_decompressor")

        # Parámetros (igual que en tu launch)
        input_topic = rospy.get_param("~compressed_image")
        output_topic = rospy.get_param("~decompressed_image")

        self.bridge = CvBridge()

        # Subscriber
        self.sub = rospy.Subscriber(
            input_topic,
            CompressedImage,
            self.callback,
            queue_size=1,
            buff_size=2**24
        )

        # Publisher
        self.pub = rospy.Publisher(
            output_topic,
            Image,
            queue_size=1
        )

        rospy.loginfo(f"Descomprimiendo {input_topic} → {output_topic}")

    def callback(self, msg):
        try:
            # Convertir bytes a array
            np_arr = np.frombuffer(msg.data, np.uint8)

            # Decodificar imagen
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Convertir a mensaje ROS Image
            ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")

            # Mantener header original
            ros_img.header = msg.header

            self.pub.publish(ros_img)

        except Exception as e:
            rospy.logerr(f"Error descomprimiendo imagen: {e}")


if __name__ == "__main__":
    ImageDecompressor()
    rospy.spin()
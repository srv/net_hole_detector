#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
import cv2
import numpy as np
from math import sin, cos, sqrt, pi
from net_hole_detector.srv import CameraSelector, CameraSelectorResponse
import yaml
import os

FRONTA_CAMERA_ID = 0
GRIPPER_CAMERA_ID = 1
RIGHT_CAMERA_ID = 2
LEFT_CAMERA_ID = 3

class ImageGeolocalizationNode:

    def __init__(self):
        rospy.init_node('image_geolocalization_node', anonymous=True)

        # Parameters (can be set via rosparam)
        self.frontal_image_topic = rospy.get_param("~frontal_original_image", "/girona500/front_camera/camera/image_raw")
        self.gripper_image_topic = rospy.get_param("~gripper_original_image", "/girona500/bravo/gripper/camera/image_raw")
        self.right_image_topic = rospy.get_param("~right_original_image", "/girona500/right_camera/camera/image_raw")
        self.left_image_topic = rospy.get_param("~left_original_image", "/girona500/left_camera/camera/image_raw")
        self.output_image_topic = rospy.get_param("~image_to_process", "/hole_detector/original_image_to_process/image_raw")

        # Camera Info
        self.frontal_camera_info_topic = rospy.get_param("~frontal_original_image", "/girona500/front_camera/camera/camera_info")
        self.gripper_camera_info_topic = rospy.get_param("~gripper_original_image", "/girona500/bravo/gripper/camera/camera_info")
        self.right_camera_info_topic = rospy.get_param("~right_original_image", "/girona500/right_camera/camera/camera_info")
        self.left_camera_info_topic = rospy.get_param("~left_original_image", "/girona500/left_camera/camera/camera_info")
        self.output_camera_info_topic = rospy.get_param("~image_to_process", "/hole_detector/original_camera_info_to_process")

        # Subscribers
        self.image_sub = rospy.Subscriber(
            self.frontal_image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1
        )
 
        self.camera_info_sub = rospy.Subscriber(
            self.frontal_camera_info_topic,
            CameraInfo,
            self.camera_info_callback,
            queue_size=1
        )
 
 
        # Publisher
        self.image_pub = rospy.Publisher(
            self.output_image_topic,
            CompressedImage,
            queue_size=10
        )

        self.camera_info_pub = rospy.Publisher(
            self.output_camera_info_topic,
            CameraInfo,
            queue_size=10
        )

        # Initialize services
        camera_topic_selector_service = rospy.Service('net_hole_detector/inference_activation_srv', CameraSelector, self.camera_topic_selector)

        rospy.loginfo("Image Geolocalization Node Initialized")
        rospy.spin()


    def image_callback(self, msg):
        # Received image published from desired camera published into output image topic
        new_msg = CompressedImage()
        new_msg.data = msg.data
        new_msg.format = msg.format
        new_msg.header = msg.header
        self.image_pub.publish(new_msg)

    def camera_info_callback(self, msg):
        # Received camera info from desired camera published into output camera info topic
        self.camera_info_pub.publish(msg)

    """

    """
    def camera_topic_selector(self, req):
        try:
            selected_id = int(req.message)
            rospy.loginfo('Selector received! Id = ' + str(selected_id))
            
            # Selected image and info camera topics
            if selected_id == FRONTA_CAMERA_ID:
                selected_image_topic = self.frontal_image_topic
                selected_camera_info_topic = self.frontal_camera_info_topic
            elif selected_id == GRIPPER_CAMERA_ID:
                selected_image_topic = self.gripper_image_topic
                selected_camera_info_topic = self.gripper_camera_info_topic
            elif selected_id == RIGHT_CAMERA_ID:
                selected_image_topic = self.right_image_topic
                selected_camera_info_topic = self.right_camera_info_topic
            elif selected_id == LEFT_CAMERA_ID:
                selected_image_topic = self.left_image_topic
                selected_camera_info_topic = self.left_camera_info_topic
            else:
                response = CameraSelectorResponse()
                response.success = False
                response.message = "Wrong camera id"
                return response 

            # Unregister old camera images topic and subscribe to new
            self.image_sub.unregister()
            self.image_sub = rospy.Subscriber(
                selected_image_topic,
                CompressedImage,
                self.image_callback,
                queue_size=1
            )

            # Unregister old camera info topic and subscribe to new
            self.camera_info_sub.unregister()
            self.camera_info_sub = rospy.Subscriber(
                selected_camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
                queue_size=1
            )
    
            # Response message
            response = CameraSelectorResponse()
            response.success = True
            response.message = "Camera selected changed"
            return response

        except Exception as e:
            rospy.logerr('Error when receiving message from service = ' + str(e))
            response = CameraSelectorResponse()
            response.success = False
            response.message = "Wrong camera id"
            return response 


if __name__ == "__main__":
    try:
        ImageGeolocalizationNode()
    except rospy.ROSInterruptException:
        pass
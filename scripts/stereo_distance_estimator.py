#!/usr/bin/env python3
import rospy
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from net_hole_detector.srv import Trigger, TriggerResponse
from statistics import mean, median
import numpy as np
from  math import pi, tan
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from net_hole_detector.msg import BoundingBox, BoundingBoxArray, Detection3D, Detection3DArray
import cv2


HYSTERESI = 50
MIN_AREA = 25
MAX_PERCENT_AREA = 0.1

class StereoDistanceEstimator:

    def __init__(self):

        self.info_topic = rospy.get_param("~camera_info_topic", "/girona500/bravo/gripper/camera/camera_info")


        rospy.Subscriber("/stereo/disparity", DisparityImage, self.callback_disparity)
        # rospy.Subscriber("/stereo/points2", PointCloud2, self.callback_distance)

        # Variables initialization
        self.__fx = 0
        self.__fy = 0
        self.__cx = 0
        self.__cy = 0

        sub1 = Subscriber("/stereo/points2", PointCloud2)
        sub2 = Subscriber("/net_hole_detector/bounding_boxes", BoundingBoxArray)
        aprox_subs = ApproximateTimeSynchronizer([sub1, sub2], queue_size=10, slop=0.1)
        aprox_subs.registerCallback(self.callback_distance)
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)


        self.pub_disp = rospy.Publisher("/stereo/disparity_image", Image, queue_size=1)
        self.pub_deb = rospy.Publisher("/stereo/deb_image", Image, queue_size=1)
        self.pub_stereo_detect3d = rospy.Publisher("net_hole_detector/stereo_detections_3d", Detection3DArray, queue_size=1)
        self.bridge = CvBridge()

        # Initialize services
        is_new_camera_selected_service = rospy.ServiceProxy('net_hole_detector/update_stereo_node_camera_info_srv', Trigger, self.__camera_change_callback)

    """
    Function: camera_change_callback

    """
    def __camera_change_callback(self, req):
        self.sub_info.unregister() 
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)
        rospy.loginfo(f"[Node] Listening to CameraInfo in: {self.info_topic}")
        
        response = TriggerResponse()
        response.success = True
        response.message = "Request processed!"
        return response 

    """
    Function: info_callback
    
    This will be executed once the bagfile sends a calibration message.
    Overwrittes any parsed YAML
    """
    def info_callback(self, msg):
        try:
            
            self.__fx = msg.K[0]
            self.__fy = msg.K[4]
            self.__cx = msg.K[2]
            self.__cy = msg.K[5]
            self.sub_info.unregister() 
            rospy.loginfo("Calibración recibida y guardada.")
        except Exception as e:
            rospy.logerr("Error - {e}")


    def callback_disparity(self, msg):
        self.pub_disp.publish(msg.image)

    def callback_distance(self, msg_point2, msg_bb):
        height = msg_point2.height
        width = msg_point2.width
        new_point_cloud = np.zeros((height, width), dtype=np.uint8)

        points = list(pc2.read_points(msg_point2, field_names=("x","y","z"), skip_nans=False))
        point_cloud = np.array(points)

        # Case with no detections
        if not msg_bb.boxes:
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

            image_msg = Image()
            image_msg = self.bridge.cv2_to_imgmsg(new_point_cloud, encoding="mono8")
            image_msg.header = msg_point2.header
            self.pub_deb.publish(image_msg)

            distance = median(parcial[~np.isnan(parcial)])

            out_msg = Detection3DArray()
            out_msg.header = msg_point2.header
            out_msg.detections = []

            det_3d = Detection3D()
            det_3d.class_id = "None"
            det_3d.score = 0
            det_3d.x = 0
            det_3d.y = 0
            det_3d.z = distance
            det_3d.width = 0
            det_3d.height = 0

            out_msg.detections.append(det_3d)

        # Case with detections
        else:
            out_msg = Detection3DArray()
            out_msg.header = msg_point2.header
            out_msg.detections = []
    
            for bb in msg_bb.boxes:
                hole_center_x = int(bb.x * width)
                hole_center_y = int(bb.y * height)

                distance = point_cloud[hole_center_x, hole_center_y, 2]
                pos_x = (hole_center_x - self.__cx) * distance / self.__fx
                pos_y = (hole_center_y - self.__cy) * distance / self.__fy

                pos_width_border_right = (hole_center_x + int(bb.width * width) - self.__cx) * distance / self.__fx
                pos_height_border_right = (hole_center_y + int(bb.height * height) - self.__cy) * distance / self.__fy

                # Create individual object
                det_3d = Detection3D()
                det_3d.class_id = bb.class_id
                det_3d.score = bb.score
                det_3d.x = pos_x
                det_3d.y = pos_y
                det_3d.z = distance
                det_3d.width = (pos_width_border_right - pos_x) * 2
                det_3d.height = (pos_height_border_right - pos_y) * 2

                out_msg.detections.append(det_3d)

        # Publish hole info gathered
        print("Distancia:" + str(distance))
        self.pub_stereo_detect3d.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("distance_reader")
    StereoDistanceEstimator()
    rospy.spin()
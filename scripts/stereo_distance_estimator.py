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

        self.info_topic = rospy.get_param("~camera_info_topic", "/girona500/xiroi/stereo_ch3/left_optical/camera_info")


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

        # We reshape the numpy array in order to use coordinates [y,x] directly
        points = list(pc2.read_points(msg_point2, field_names=("x","y","z"), skip_nans=False))
        point_cloud = np.array(points).reshape((height, width, 3))

        # This is to create a debug image
        new_point_cloud = np.zeros((height, width), dtype=np.uint8)


        # ----- CASE WITH NO DETECTIONS -----
        if not msg_bb.boxes:
            print('Without detection!')
            # Simplified: Looking for a square in the image center
            h_start, h_end = height//2 - HYSTERESI, height//2 + HYSTERESI
            w_start, w_end = width//2 - HYSTERESI, width//2 + HYSTERESI

            # Extract Z layer (index 2)
            central_zone_z = point_cloud[h_start:h_end, w_start:w_end, 2]

            # Filtering NaNs and absurd distances 
            mask = (~np.isnan(central_zone_z)) & (central_zone_z > 0.05) & (central_zone_z < 8.0)
            valid_depths = central_zone_z[mask]

            if valid_depths.size > 0:
                distance = np.median(valid_depths)
                z_min, z_max = np.min(valid_depths), np.max(valid_depths)
                rospy.loginfo(f"[DEBUG NO-DET] Centro Imagen | Z-Median: {distance:.3f}m (Min: {z_min:.3f}, Max: {z_max:.3f}) | Pixels: {len(valid_depths)}")
            
            else:
                distance = 0.0
                rospy.logwarn("[DEBUG NO-DET] Centro Imagen sin datos válidos")

            # Drawing debug square
            new_point_cloud[h_start:h_end, w_start:w_end] = 255

        # ----- CASE WITH DETECTIONS -----
        else:
            print('Detection!')
            out_msg = Detection3DArray()
            out_msg.header = msg_point2.header
            out_msg.detections = []

            for bb in msg_bb.boxes:
                # Center coordinates in pixels
                u1 = int((bb.x - bb.w/2) * width)
                u2 = int((bb.x + bb.w/2) * width)
                v1 = int((bb.y - bb.h/2) * height)
                v2 = int((bb.y + bb.h/2) * height)

                # Avoid getting out of bound
                u1, u2 = max(0, u1), min(width-1, u2)
                v1, v2 = max(0, v1), min(height-1, v2)

                # 2. Dibujamos la BBox en la imagen de debug (Topic Blanco y Negro)
                # Dibujamos los bordes de la caja con valor 255 (blanco)
                new_point_cloud[v1:v2, u1] = 255 # Línea izquierda
                new_point_cloud[v1:v2, u2] = 255 # Línea derecha
                new_point_cloud[v1, u1:u2] = 255 # Línea superior
                new_point_cloud[v2, u1:u2] = 255 # Línea inferior


                # For every pixel of the image, we have the 3 data numbers: the real coordinates X,Y,Z
                # In numpy: [Row (Y), Column (X)]
                z_array = point_cloud[v1:v2, u1:u2, 2]

                # Filtering NaNs and distances (more than 8 meters is an error)
                mask = (~np.isnan(z_array)) & (z_array> 0.05) & (z_array < 8.0)
                valid_depths = z_array[mask]

                if valid_depths.size > 0:
                    # We use the median to ignore 'outliers'
                    distance = np.median(valid_depths)
                    z_min, z_max = np.min(valid_depths), np.max(valid_depths)
                    
                    # LOG CRÍTICO: ¿Qué está viendo realmente la cámara en el agujero?
                    rospy.loginfo(f"[DEBUG DET] AGUJERO | Pixels: u({u1}-{u2}) v({v1}-{v2})")
                    rospy.loginfo(f"[DEBUG DET] Z-Depth | Median: {distance:.3f}m (Min: {z_min:.3f}, Max: {z_max:.3f}) | Count: {len(valid_depths)}")
                else:
                    distance = float('nan')
                    rospy.logwarn("[DEBUG DET] Agujero detectado pero sin profundidad válida")


                # 4. Calculamos X e Y usando la distancia encontrada
                # Usamos el centro de la BBox para la posición 3D
                hole_center_x_px = int(bb.x * width)
                hole_center_y_px = int(bb.y * height)

                if not np.isnan(distance):
                    pos_x = (hole_center_x_px - self.__cx) * distance / self.__fx
                    pos_y = (hole_center_y_px - self.__cy) * distance / self.__fy

                    # LOG DE PROYECCIÓN: Coordenadas relativas a la cámara izquierda
                    rospy.loginfo(f"[DEBUG DET] 3D Cam Frame: X={pos_x:.3f}, Y={pos_y:.3f}, Z={distance:.3f}")

                    det_3d = Detection3D()
                    det_3d.class_id = bb.class_id
                    det_3d.score = bb.score
                    det_3d.x, det_3d.y, det_3d.z = pos_x, pos_y, distance

                    # Real Size (meters) calculation based on box size 
                    det_3d.width = (bb.w * width * distance) / self.__fx
                    det_3d.height = (bb.h * height * distance) / self.__fy

                    out_msg.detections.append(det_3d)

            self.pub_stereo_detect3d.publish(out_msg)

        # Publish debug image using cv_bridge
        image_msg = self.bridge.cv2_to_imgmsg(new_point_cloud, encoding = "mono8")
        image_msg.header = msg_point2.header
        self.pub_deb.publish(image_msg)

        print(f"'Real' distance to center: {distance:.3f} meters")


        # if not msg_bb.boxes:
        #     i_height = int(height / 2) - int(HYSTERESI)
        #     i_width = int(width / 2)
        #     for i in range(HYSTERESI*2):         

        #         punt_1 = int(height * (width/2 - HYSTERESI + i) + (width/2 - HYSTERESI) - 1)
        #         punt_2 = int(height * (width/2 - HYSTERESI + i) + (width/2 + HYSTERESI) - 1)

        #         if i == 0:
        #             parcial = point_cloud[punt_1:punt_2, 2]
        #         else:
        #             parcial = np.append(parcial, point_cloud[punt_1:punt_2, 2])
        #         new_point_cloud[i_height, i_width-HYSTERESI:i_width+HYSTERESI] = 255
        #         i_height += 1

        #     image_msg = Image()
        #     image_msg = self.bridge.cv2_to_imgmsg(new_point_cloud, encoding="mono8")
        #     image_msg.header = msg_point2.header
        #     self.pub_deb.publish(image_msg)

        #     distance = median(parcial[~np.isnan(parcial)])

        #     out_msg = Detection3DArray()
        #     out_msg.header = msg_point2.header
        #     out_msg.detections = []

        #     det_3d = Detection3D()
        #     det_3d.class_id = "None"
        #     det_3d.score = 0
        #     det_3d.x = 0
        #     det_3d.y = 0
        #     det_3d.z = distance
        #     det_3d.width = 0
        #     det_3d.height = 0

        #     out_msg.detections.append(det_3d)

        # # ----- CASE WITH DETECTIONS -----
        # else:
        #     out_msg = Detection3DArray()
        #     out_msg.header = msg_point2.header
        #     out_msg.detections = []
    
        #     for bb in msg_bb.boxes:
        #         hole_center_x = int(bb.x * width)
        #         hole_center_y = int(bb.y * height)

        #         distance = point_cloud[hole_center_x, hole_center_y, 2]
        #         pos_x = (hole_center_x - self.__cx) * distance / self.__fx
        #         pos_y = (hole_center_y - self.__cy) * distance / self.__fy

        #         pos_width_border_right = (hole_center_x + int(bb.width * width) - self.__cx) * distance / self.__fx
        #         pos_height_border_right = (hole_center_y + int(bb.height * height) - self.__cy) * distance / self.__fy

        #         # Create individual object
        #         det_3d = Detection3D()
        #         det_3d.class_id = bb.class_id
        #         det_3d.score = bb.score
        #         det_3d.x = pos_x
        #         det_3d.y = pos_y
        #         det_3d.z = distance
        #         det_3d.width = (pos_width_border_right - pos_x) * 2
        #         det_3d.height = (pos_height_border_right - pos_y) * 2

        #         out_msg.detections.append(det_3d)

        # # Publish hole info gathered
        # print("Distancia:" + str(distance))
        # self.pub_stereo_detect3d.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("distance_reader")
    StereoDistanceEstimator()
    rospy.spin()
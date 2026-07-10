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
        sub2 = Subscriber("yolo/detections", BoundingBoxArray)
        aprox_subs = ApproximateTimeSynchronizer([sub1, sub2], queue_size=10, slop=0.1)
        aprox_subs.registerCallback(self.callback_distance)
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)


        self.pub_disp = rospy.Publisher("/stereo/disparity_image", Image, queue_size=1)
        self.pub_deb = rospy.Publisher("/stereo/deb_image", Image, queue_size=1)
        self.pub_stereo_detect3d = rospy.Publisher("yolo/stereo_detections_3d", Detection3DArray, queue_size=1)
        self.bridge = CvBridge()

    """
    Function: info_callback
    
    This will be executed once the bagfile sends a calibration message.
    Overwrittes any parsed YAML
    """
    def info_callback(self, msg: CameraInfo):
        try:
            
            self.__fx = msg.K[0]
            self.__fy = msg.K[4]
            self.__cx = msg.K[2]
            self.__cy = msg.K[5]
            self.sub_info.unregister() 
            rospy.loginfo("Calibración recibida y guardada.")
        except Exception as e:
            rospy.logerr("Error - {e}")


    def callback_disparity(self, msg: DisparityImage):
        self.pub_disp.publish(msg.image)

    def callback_distance(self, msg_point2: PointCloud2, msg_bb: BoundingBoxArray):
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
                hole_center_x_px = int(bb.corr_x * width)
                hole_center_y_px = int(bb.corr_y * height)

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
                    det_3d.is_corrosion = True
                    det_3d.area_px = bb.area
                    det_3d.area_real = bb.area * distance**2 / (self.__fx * self.__fy)

                    out_msg.detections.append(det_3d)

            self.pub_stereo_detect3d.publish(out_msg)

        # Publish debug image using cv_bridge
        image_msg = self.bridge.cv2_to_imgmsg(new_point_cloud, encoding = "mono8")
        image_msg.header = msg_point2.header
        self.pub_deb.publish(image_msg)

        print(f"'Real' distance to center: {distance:.3f} meters")



if __name__ == "__main__":
    rospy.init_node("distance_reader")
    StereoDistanceEstimator()
    rospy.spin()
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

# ----- ROS MESSAGES -----
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped

# ----- OWN CLASSES -----
from net_hole_detector.camera_geometry import CameraGeometry
from net_hole_detector.scale_estimator import ScaleEstimator

class NetHoleDetectorNode:
    def __init__(self):
        rospy.init_node('net_hole_detector')

        # ==========================================
        # 1. CONFIGURATION & PARAMETERS (From Launch)
        # ==========================================

        # We read this topics from the launch file (and we have default ones)
        default_image_topic = "/girona500/bravo/gripper/camera/image_rect/compressed"
        self.image_topic = rospy.get_param("~image_topic", default_image_topic)
        default_info_topic = "/girona500/bravo/gripper/camera/camera_info"
        self.info_topic = rospy.get_param("~camera_info_topic", default_info_topic)
        self.yolo_topic = rospy.get_param("~yolo_topic", "/yolo/output_raw")
        
        # We try to load a Calibration YAML file (In case CameraInfo topic does not exist)
        default_yaml = rospy.get_param("~calibration_file", "") 

        # Frame ID (In which coordinates system is the point)
        self.camera_frame = rospy.get_param("~camera_frame", "bravo_camera_optical_frame")
        
        # to use default camera (gripper):
        # roslaunch net_hole_detector detector.launch
        # to use another camera (laser, ...):
        # roslaunch net_hole_detector detector.launch image_topic_name:=/girona500/down_camera/camera/image_raw

        # ==========================================
        # 2. CLASSES INITIALIZATION
        # ==========================================

        # CameraGeometry: Tries to parser a YAML if it exist, otherwise, it initializes empty
        self.geo = CameraGeometry(yaml_path=default_yaml if default_yaml else None)
        
        # Estimator: Blobs Logic to get Z
        self.estimator = ScaleEstimator()

        # OpenCV-ROS Bridge
        self.bridge = CvBridge()

        # ==========================================
        # 3. STATE VARIABLES (MEMORY)
        # ==========================================

        # Store the "Z" calculated by the last image
        # We store the time of this last calculation to avoid lagging
        self.latest_z = None
        # Initialize with time 0 so the first check fails until we get a real image
        self.last_z_time = rospy.Time(0)
        
        # Control Flags
        # If the name of the topic contains "rect", we assume we do not need to correct distorsion
        if "rect" in self.image_topic:
            self.is_already_rectified = True
            rospy.loginfo(f"[Node] Topic '{self.image_topic}' detected as RECTIFIED. Mode: PASSTHROUGH.")
        else:
            self.is_already_rectified = False
            rospy.loginfo(f"[Node] Topic '{self.image_topic}' detected as RAW. Mode: UNDISTORT active.")
        
        # ==========================================
        # 4. SUBSCRIPTIONS
        # ==========================================

        # A) Camera INFO (To update fx, fy, cx, cy automatically)
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)
        rospy.loginfo(f"[Node] Listening to CameraInfo in: {self.info_topic}")

        # B) IMAGE (To calculate Global Z)
        if "compressed" in self.image_topic:
            self.sub_img = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1)
        else:
            self.sub_img = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo(f"[Node] Listening to Images in: {self.image_topic}")

        # C) YOLO (To calculate X and Y of the specific hole)
        # Expecting array: [class_id, x_center_norm, y_center_norm, w_norm, h_norm]
        self.sub_yolo = rospy.Subscriber(self.yolo_topic, Float32MultiArray, self.yolo_callback)
        rospy.loginfo(f"[Node] Listening to YOLO in: {self.yolo_topic}")

        # ==========================================
        # 5. PUBLICADORES
        # ==========================================

        # We create a topic where to publish the final result: the 3D position of the hole
        self.pub_point = rospy.Publisher("net_hole_detector/target_point", PointStamped, queue_size=1)
    

    # =========================================================
    # CALLBACK 1: UPDATE CALIBRATION
    # =========================================================
    """
    Function: info_callback
    
    """
    def info_callback(self, msg):
        """
        This will be executed once the bagfile sends a calibration message.
        Overwrittes any parsed YAML
        """
        self.geo.set_camera_info(msg)
        # If we are SURE camera calibration does not vary, we can stop listening to save CPU
        # self.sub_info.unregister() 
        # rospy.loginfo("Calibración recibida y guardada. Desuscribiendo del topic de info.")


    # =========================================================
    # CALLBACK 2: PROCESS IMAGE -> CALCULATE 'Z'
    # =========================================================
    """
    Function: image_callback

    """
    def image_callback(self, msg):
        # 1. Decode Image
        try:
            if hasattr(msg, 'format') and "compressed" in str(type(msg)):
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error decodificando imagen: {e}")
            return
        
        # SAFETY CHECK: If decoding returns None (corrupted data), we stop
        if cv_img is None:
            return

        # 2. Update REAL Dimensions (in case they have change, we get the ones from the real image)
        h, w = cv_img.shape[:2]
        self.geo.img_h = h
        self.geo.img_w = w

        # 3. Smart Pre-Processing (Undistort)
        # If it is NOT rectified and we have calibration data -> We Correct
        if not self.is_already_rectified and self.geo.is_calibrated:
            img_process = self.geo.undistort_image(cv_img)
        else:
            # If is already rect or we do not have calibration (nor yaml, nor topic), using the original
            img_process = cv_img

        # 4. Scale Estimator
        # Detection blobs, area calculation... logic
        scale = self.estimator.get_scale_from_blobs(img_process)
        
        if scale:
            # ¡SUCCESS! We store (update) Z and the time in the Memory of the class
            self.latest_z = self.geo.get_z_distance(scale)
            self.last_z_time = rospy.Time.now()
            # rospy.loginfo(f"Estimated distance: {z:.2f} m")
        else:
            # If we do not see the net, we preserve the last known Z
            pass


    # =========================================================
    # CALLBACK 3: PROCESAR YOLO -> CALCULAR 'X, Y' Y PUBLICAR
    # =========================================================
    """
    Function: yolo_callback
    """
    def yolo_callback(self, msg):
        """
        It executes everytime YOLO sees something
        msg.data structure [class_id, x_norm, y_norm, w_norm, h_norm]
        """
        # 1. Sanity Checks
        # Existance
        if self.latest_z is None:
            rospy.logwarn_throttle(2, "[Yolo] Object detected, but Z distance unkwown (waiting for image...)")
            return
        
        # Check Timeout (Caducity)
        time_diff = rospy.Time.now() - self.last_z_time

        # If it is 0.5s or older, its dangerous
        if time_diff.to_sec() > 0.5:
            rospy.logwarn_throttle(2, f"[Yolo] Z data too old ({time_diff.to_sec():.2f}s). Ignoring Yolo to avoid mistakes.")
            return
        
        # Calibration
        if not self.geo.is_calibrated:
            rospy.logwarn_throttle(2, "[Yolo] Unable to get Camera calibration. Impossible to project 3D")
            return

        # 2. Handle YOLO data
        # msg.data is the list (tuple) [class, x, y, w, h] that is expected
        yolo_bbox = msg.data

        # Transform 0-1 -> Pixels (u, v)
        u, v = self.geo.yolo_to_pixels(yolo_bbox)

        # Project from 2D -> 3D (X, Y, Z)
        point_3d_array = self.geo.project_pixel_to_3d(u, v, self.latest_z)

        # Check for error (0,0,0)
        if np.all(point_3d_array == 0) and self.latest_z != 0:
            rospy.logwarn_throttle(2, "Error projecting a 3D (Calibration is missing)")
            return
        
        # Unpack numpy array
        x_meters, y_meters, z_meters = point_3d_array

        # 3. Publish ruslt (PointStamped)
        point_msg = PointStamped()
        
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = self.camera_frame
        
        point_msg.point.x = x_meters
        point_msg.point.y = y_meters
        point_msg.point.z = z_meters

        self.pub_point.publish(point_msg)
        
        rospy.loginfo(f"TARGET 3D: X={x_meters:.3f} Y={y_meters:.3f} Z={z_meters:.3f}")


if __name__ == '__main__':
    try:
        node = NetHoleDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
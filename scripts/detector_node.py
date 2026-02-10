#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

# ----- ROS MESSAGES -----
from sensor_msgs.msg import Image, CameraInfo

# ----- CUSTOM MESSAGES -----
from net_hole_detector.msg import BoundingBox, BoundingBoxArray
from net_hole_detector.msg import Detection3D, Detection3DArray

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
        self.image_topic = rospy.get_param("~image_topic", "/girona500/bravo/gripper/camera/image_rect/compressed")
        self.info_topic = rospy.get_param("~camera_info_topic", "/girona500/bravo/gripper/camera/camera_info")
        self.yolo_topic = rospy.get_param("~yolo_topic", "/yolo/detections")
        
        # Safety Limit: 5 Detections per message at max.
        self.max_detections = rospy.get_param("~max_detections", 5)
        
        # We try to load a Calibration YAML file (In case CameraInfo topic does not exist)
        default_yaml = rospy.get_param("~calibration_file", "") 
        
        # ==========================================
        # ¡¡¡¡¡¡¡¡¡¡¡¡¡¡CHECK NAME!!!!!!!!!!!!!!!!!!
        # ==========================================
        # Frame ID (In which coordinates system is the point)
        self.camera_frame = rospy.get_param("~camera_frame", "bravo_camera_optical_frame")

        # ==========================================
        # 2. CONFIGURACIÓN DE MODO (RECT vs RAW)
        # ==========================================

        # Control Flags based on topic name
        # If the name of the topic contains "rect", we assume we do not need to correct distorsion
        if "rect" in self.image_topic:
            self.is_already_rectified = True
            rospy.loginfo(f"[Node] Topic '{self.image_topic}' detected as RECTIFIED. Mode: PASSTHROUGH.")
        else:
            self.is_already_rectified = False
            rospy.loginfo(f"[Node] Topic '{self.image_topic}' detected as RAW. Mode: UNDISTORT active.")

        # ==========================================
        # 3. CLASSES INITIALIZATION
        # ==========================================

        # CameraGeometry: Tries to parser a YAML if it exist, otherwise, it initializes empty
        self.geo = CameraGeometry(yaml_path=default_yaml if default_yaml else None)
        
        # Once the YAML is loaded, we force the Right Matrix Selection
        # If we do not force it, it would use always the K Matrix by default
        if self.geo.is_calibrated:
            self.geo.select_matrix(self.is_already_rectified)
        
        # Estimator: Blobs Logic to get Z
        self.estimator = ScaleEstimator()

        # OpenCV-ROS Bridge
        self.bridge = CvBridge()

        # ==========================================
        # 4. STATE VARIABLES (MEMORY)
        # ==========================================

        # Store the "Z" calculated by the last image
        # We store the time of this last calculation to avoid lagging
        self.latest_z = None
        # Initialize with time 0 so the first check fails until we get a real image
        self.last_z_time = rospy.Time(0)
        
        # ==========================================
        # 5. SUBSCRIPTIONS
        # ==========================================

        # A) Camera INFO (To update fx, fy, cx, cy automatically)
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)
        rospy.loginfo(f"[Node] Listening to CameraInfo in: {self.info_topic}")

        # B) IMAGE (To calculate Global Z)
        # ¡¡¡ALWAYS WILL RECEIVE RAW!!!
        self.sub_img = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo(f"[Node] Listening to Images in: {self.image_topic}")

        # C) YOLO (To calculate X and Y of the specific hole)
        # Expecting BoundingBoxArray type message
        self.sub_yolo = rospy.Subscriber(self.yolo_topic, BoundingBoxArray, self.yolo_callback)
        rospy.loginfo(f"[Node] Listening to YOLO in: {self.yolo_topic}")

        # ==========================================
        # 6. PUBLICADORES
        # ==========================================

        # We create a topic where to publish the final result: the 3D position of the hole
        self.pub_point = rospy.Publisher("net_hole_detector/detections_3d", Detection3DArray, queue_size=1)
    

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

        # [CRITICAL CORRECTION]
        # Every Time new info arrives, we ensure fx, fy, cx, cy actualizes according to our mode (Rect vs Raw)
        self.geo.select_matrix(self.is_already_rectified)

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
        scale = self.estimator.get_scale_from_blobs(img_process, real_area_m2=(0.015*0.015))
        
        if scale:
            # ¡SUCCESS! We store (update) Z and the time in the Memory of the class
            self.latest_z = self.geo.get_z_distance(scale)
            self.last_z_time = rospy.Time.now()
            # rospy.loginfo(f"Estimated distance: {z:.2f} m")
        else:
            # If we do not see the net, we preserve the last known Z
            pass


    # =========================================================
    # CALLBACK 3: PROCESS YOLO -> CALCULATE 'X, Y' AND PUBLISH
    # =========================================================
    """
    Function: yolo_callback
    """
    def yolo_callback(self, msg):
        """
        It executes everytime YOLO sees something
        """
        # 1. Sanity Checks
        # Existance
        if len(msg.boxes) == 0:
            return # No hay nada que detectar
        
        if self.latest_z is None:
            rospy.logwarn_throttle(2, "[Yolo] Object detected, but Z distance unkwown (waiting for image...)")
            return
        
        # 2. "TOP-K" FILTER: Sort and Cut
        # a) Sort all bboxes by score 
        #    So we stay with the 'good' ones.
        all_boxes_sorted = sorted(msg.boxes, key=lambda b: b.score, reverse=True)

        # b) Aply the cut.
        best_boxes = all_boxes_sorted[:self.max_detections]

        # (Opcional) INFO Log: we have noise
        if len(msg.boxes) > self.max_detections:
            rospy.logdebug(f"Active Filter: {len(msg.boxes)} bboxes received, sending Top-{self.max_detections}")

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

        # 3. Prepare output message
        out_msg = Detection3DArray()
        out_msg.header = msg.header
        out_msg.detections = []

        # 4. Iterate over BEST bboxes:
        for box in best_boxes:
            # box is and object (pose, dimensions, ...)
            # We expect a list: [label, x, y, w, h]

            # Extract data from ROS message
            bbox_list = [
                box.class_id,  # Posición 0
                box.x,         # Posición 1 (x_n)
                box.y,         # Posición 2 (y_n)
                box.w,         # Posición 3
                box.h          # Posición 4
            ]

            try:
                # Transform 0-1 -> Pixels (u, v)
                u, v = self.geo.yolo_to_pixels(bbox_list)

                # Project from 2D -> 3D (X, Y, Z)
                point_3d = self.geo.project_pixel_to_3d(u, v, self.latest_z)

                if hasattr(point_3d, '__len__'):
                    # Create individual object
                    det_3d = Detection3D()
                    det_3d.class_id = box.class_id
                    det_3d.score = box.score
                    det_3d.x = point_3d[0]
                    det_3d.y = point_3d[1]
                    det_3d.z = point_3d[2]

                    # Add to the list
                    out_msg.detections.append(det_3d)
            
            except Exception as e:
                rospy.logwarn(f"Error processing detection: {e}")

        # 5. Publish list
        self.pub_point.publish(out_msg)
        
        # Useful Info
        rospy.loginfo_throttle(2, f"Publicadas {len(out_msg.detections)} detecciones 3D (Max config: {self.max_detections}). Z ref: {self.latest_z:.2f}m")


if __name__ == '__main__':
    try:
        node = NetHoleDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
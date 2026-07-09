#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Float32

# Parámetros por defecto
BASELINE = 0.10 # Distancia entre cámaras en metros
EPIPOLAR_TOLERANCE = 25.0 # Tolerancia en píxeles para el Plan A
MIN_MATCHES_PLAN_A = 10 # Si caemos por debajo de esto, activamos Plan B

class SparseStereoMeshEstimator:
    def __init__(self):
        rospy.init_node('sparse_stereo_mesh')

        # Tópicos
        self.left_img_topic = rospy.get_param("~left_img_topic", "/girona500/xiroi/stereo_ch3/left_optical/image_color")
        self.right_img_topic = rospy.get_param("~right_img_topic", "/girona500/xiroi/stereo_ch3/right_optical/image_color")
        self.left_info_topic = rospy.get_param("~left_info_topic", "/girona500/xiroi/stereo_ch3/left_optical/camera_info")
        self.right_info_topic = rospy.get_param("~right_info_topic", "/girona500/xiroi/stereo_ch3/right_optical/camera_info")

        # Variables intrínsecas (solo necesitamos las de la izquierda y fx)
        self.__fx = 0
        self.__fy = 0
        self.__cx = 0
        self.__cy = 0
        self.calibrated = False

        # Herramientas
        self.bridge = CvBridge()
        self.sift = cv2.SIFT_create()
        # Usamos FLANN para que el matching sea más rápido
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Publicadores
        self.pub_points = rospy.Publisher("/net_hole_detector/net_mesh_3d", PointCloud2, queue_size=1)
        self.pub_debug = rospy.Publisher("/stereo/sift_matches_debug", Image, queue_size=1)
        self.pub_sparse_dist = rospy.Publisher("/net_hole_detector/sparse_distance", Float32, queue_size=1)

        # Suscriptores de calibración (se desuscriben tras recibir el primer mensaje)
        self.sub_left_info = rospy.Subscriber(self.left_info_topic, CameraInfo, self.info_callback)

        # Sincronizador de imágenes
        sub_left_img = Subscriber(self.left_img_topic, Image)
        sub_right_img = Subscriber(self.right_img_topic, Image)
        self.ts = ApproximateTimeSynchronizer([sub_left_img, sub_right_img], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.stereo_callback)

        rospy.loginfo("Sparse Stereo Mesh Estimator inicializado. Esperando imágenes...")

    def info_callback(self, msg):
        """ Recibe la matriz K de la cámara izquierda y la guarda """
        try:
            self.__fx = msg.K[0]
            self.__fy = msg.K[4]
            self.__cx = msg.K[2]
            self.__cy = msg.K[5]
            self.calibrated = True
            self.sub_left_info.unregister()
            rospy.loginfo(f"Calibración estéreo adquirida: fx={self.__fx:.2f}, cx={self.__cx:.2f}, cy={self.__cy:.2f}")
        except Exception as e:
            rospy.logerr(f"Error parseando CameraInfo: {e}")

    def stereo_callback(self, left_msg, right_msg):
        if not self.calibrated:
            return

        # 1. Convertir imágenes de ROS a OpenCV
        try:
            img_l = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
            img_r = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error de CvBridge: {e}")
            return
        
        # 2. Extracción de Keypoints y Descriptores con SIFT
        kp_l, des_l = self.sift.detectAndCompute(img_l, None)
        kp_r, des_r = self.sift.detectAndCompute(img_r, None)

        #rospy.loginfo("--------------------------------------------------")
        #rospy.loginfo(f"[FASE 1] Keypoints encontrados por SIFT: Izquierda={len(kp_l) if kp_l else 0}, Derecha={len(kp_r) if kp_r else 0}")

        if des_l is None or des_r is None or len(kp_l) < 2 or len(kp_r) < 2:
            rospy.logwarn("Aviso: SIFT no encontró suficientes puntos en las imágenes.")
            return

        # 3. Matching inicial usando Lowe's Ratio Test
        matches = self.matcher.knnMatch(des_l, des_r, k=2)
        #rospy.loginfo(f"[FASE 2] Matches brutos iniciales (KNN): {len(matches)}")
        good_matches = []
        for match_k in matches:
            if len(match_k) == 2:
                m, n = match_k
                if m.distance < 0.85 * n.distance:
                    good_matches.append(m)

        #rospy.loginfo(f"[FASE 3] Matches tras Filtro de Lowe (0.95): {len(good_matches)}")

        # 4. Filtrado (Plan A vs Plan B)
        valid_matches = []

        # --- PLAN A: Filtrado Geométrico Rápido (Línea Epipolar) ---
        for m in good_matches:
            pt_l = kp_l[m.queryIdx].pt
            pt_r = kp_r[m.trainIdx].pt
            y_diff = abs(pt_l[1] - pt_r[1])
            x_diff = pt_l[0] - pt_r[0] # La disparidad debe ser positiva (cámara der está a la derecha)
            
            if y_diff <= EPIPOLAR_TOLERANCE and x_diff > 0:
                valid_matches.append(m)

        #rospy.loginfo(f"[FASE 4] Matches Válidos tras Plan A (Tolerancia 15px): {len(valid_matches)}")

        # # --- PLAN B: Paracaídas RANSAC (Si el Plan A falla estrepitosamente) ---
        # if len(valid_matches) < MIN_MATCHES_PLAN_A and len(good_matches) > 10:
        #     rospy.logwarn("[PLAN B ACTIVADO] Posible descalibración. Calculando Matriz Fundamental...")
        #     pts_l = np.float32([kp_l[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        #     pts_r = np.float32([kp_r[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
        #     # Buscamos la relación real de las cámaras ahora mismo
        #     F, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, 3.0, 0.99)
            
        #     if mask is not None:
        #         valid_matches = []
        #         mask_list = mask.ravel().tolist()
        #         for i, is_inlier in enumerate(mask_list):
        #             if is_inlier:
        #                 valid_matches.append(good_matches[i])

        # 5. Cálculo Espacial 3D (Triangulación)
        cloud_points = []
        for m in valid_matches:
            pt_l = kp_l[m.queryIdx].pt
            pt_r = kp_r[m.trainIdx].pt
            
            disparity = pt_l[0] - pt_r[0]
            if disparity <= 0.1: # Evitar división por cero
                continue
                
            # Fórmulas que ya usabas
            Z = (self.__fx * BASELINE) / disparity
            X = (pt_l[0] - self.__cx) * Z / self.__fx
            Y = (pt_l[1] - self.__cy) * Z / self.__fy
            
            # Filtramos puntos locos (ej. más allá de 8 metros o menos de 0.1)
            if 0.1 < Z < 8.0:
                cloud_points.append([X, Y, Z])


            # Filtrar tots els punts que estiguin a més de 40 vegades el baseline.
        #rospy.loginfo(f"[FASE 5] Puntos 3D reales publicados en la Nube: {len(cloud_points)}")

        # 6. Publicación de Resultados
        # Publicar Nube de Puntos de la red
        if cloud_points:
            header = Header()
            header.stamp = left_msg.header.stamp
            header.frame_id = left_msg.header.frame_id # Usualmente algo como "left_optical_frame"
            pc2_msg = pc2.create_cloud_xyz32(header, cloud_points)
            self.pub_points.publish(pc2_msg)

        # Publicar Imagen de Debug
        if self.pub_debug.get_num_connections() > 0:
            img_matches = cv2.drawMatches(img_l, kp_l, img_r, kp_r, valid_matches, None, 
                                          matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=0)
            msg_debug = self.bridge.cv2_to_imgmsg(img_matches, "bgr8")
            msg_debug.header = left_msg.header
            self.pub_debug.publish(msg_debug)

        # Publicar distancia estimada por SIFT
        if cloud_points:
            z_cords = [p[2] for p in cloud_points]
            median_z = np.median(z_cords)
            self.pub_sparse_dist.publish(Float32(median_z))
        else:
            self.pub_sparse_dist.publish(Float32(float('nan')))

if __name__ == '__main__':
    try:
        SparseStereoMeshEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
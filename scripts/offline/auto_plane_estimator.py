#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import yaml
import numpy as np
import tf2_ros
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans

class AutoPlaneEstimator:
    def __init__(self):
        rospy.init_node('auto_plane_estimator', anonymous=True)

        # ---- Parámetros Modulares ----
        self.yaml_path = rospy.get_param("~yaml_path", "")
        self.target_frame = rospy.get_param("~target_frame", "world_ned")
        self.sample_interval = rospy.get_param("~sample_interval", 5.0)  # Segundos entre capturas
        self.timeout_seconds = rospy.get_param("~timeout_seconds", 6.0)  # Segundos de silencio para acabar
        self.ransac_threshold = rospy.get_param("~ransac_threshold", 0.1) # Tolerancia RANSAC en metros

        if not self.yaml_path:
            rospy.logerr("[RANSAC] ¡Debes especificar el parámetro yaml_path!")
            rospy.signal_shutdown("Missing parameter")
            return

        # ---- Variables de control ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.all_points_world = [] # Lista maestra de puntos [X, Y, Z]
        
        self.last_sample_time = None
        self.last_msg_time = None
        self.is_processing = False

        # Suscriptor a la nube de puntos
        self.sub_pc = rospy.Subscriber("/stereo/points2", PointCloud2, self.cloud_callback)
        
        # Temporizador para el "Detector de Silencio" (Se ejecuta a 1 Hz)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.check_timeout)

        rospy.loginfo(f"[RANSAC] Iniciado. Muestreo cada {self.sample_interval}s. Finalizará tras {self.timeout_seconds}s de silencio.")

    def cloud_callback(self, msg):
        if self.is_processing:
            return

        # Actualizamos la hora del último mensaje recibido para el detector de silencio
        self.last_msg_time = rospy.Time.now()

        # Usamos el stamp del propio mensaje para el control de intervalos (más preciso en rosbags)
        msg_time = msg.header.stamp.to_sec()

        if self.last_sample_time is None:
            self.last_sample_time = msg_time
            rospy.loginfo("[RANSAC] ¡Primer frame recibido! Guardando y empezando el muestreo...")
        else:
            # Comprobamos si ha pasado el intervalo deseado
            if (msg_time - self.last_sample_time) < self.sample_interval:
                return # Ignoramos este frame, aún no toca
            
            self.last_sample_time = msg_time
            rospy.loginfo(f"[RANSAC] Capturando frame (Acumulados: {len(self.all_points_world) + 1})")

        # ---- Transformación TF ----
        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame, 
                msg.header.frame_id, 
                msg.header.stamp, 
                rospy.Duration(0.1)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"[RANSAC] TF no disponible para este frame: {e}")
            return

        # Extraer matriz de rotación y traslación
        q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
        t = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
        R_mat_4x4 = tf_trans.quaternion_matrix(q)
        R_mat = R_mat_4x4[:3, :3] # Nos quedamos con la submatriz 3x3 de rotación pura

        # Extraer puntos y limpiar NaNs
        points_local = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        
        if points_local.size == 0:
            return

        # Transformación matricial a World NED
        points_world = (R_mat @ points_local.T).T + t
        self.all_points_world.append(points_world)

    def check_timeout(self, event):
        """ Se ejecuta en bucle para comprobar si la bolsa ha terminado """
        if self.is_processing or self.last_msg_time is None:
            return
            
        current_time = rospy.Time.now()
        silence_duration = (current_time - self.last_msg_time).to_sec()
        
        if silence_duration > self.timeout_seconds:
            rospy.logwarn(f"[RANSAC] Detectados {silence_duration:.1f}s de silencio. ¡Asumiendo fin del bagfile!")
            self.is_processing = True
            self.sub_pc.unregister() # Cortamos el grifo de datos
            self.timer.shutdown()
            self.process_and_save()

    def process_and_save(self):
        rospy.loginfo("[RANSAC] Iniciando cálculo geométrico...")
        
        if not self.all_points_world:
            rospy.logerr("[RANSAC] No se acumularon puntos válidos. Revisa el TF y la bolsa.")
            rospy.signal_shutdown("No data")
            return

        # Juntar todos los bloques capturados en una super-matriz Nx3
        pts_3d_dense = np.vstack(self.all_points_world)
        num_pts_densos = pts_3d_dense.shape[0]
        rospy.loginfo(f"[RANSAC] Puntos densos acumulados en RAM: {num_pts_densos}")

        # =====================================================================
        # 🔥 LAS 3 LÍNEAS MÁGICAS: Reducimos drásticamente para acelerar la CPU
        # Cogemos 1 de cada 200 puntos (Ajusta el 200 si quieres más o menos puntos)
        # =====================================================================
        stride = max(1, num_pts_densos // 100000) # Forzamos que se quede rondando los 100.000 puntos máx.
        pts_3d = pts_3d_dense[::stride]
        rospy.loginfo(f"[RANSAC] Puntos aligerados (subsampled con paso {stride}): {pts_3d.shape[0]}")
        # =====================================================================

        # Proyección 2D (Norte-Este)
        pts_xy = pts_3d[:, :2]

        # ---- ALGORITMO RANSAC 2D ----
        best_inliers = []
        best_line = None
        max_iterations = 2000 # Más iteraciones porque la nube será densa
        
        num_pts = pts_xy.shape[0]
        if num_pts < 2:
            rospy.signal_shutdown("Insufficient points")
            return

        for _ in range(max_iterations):
            idx = np.random.choice(num_pts, 2, replace=False)
            p1, p2 = pts_xy[idx[0]], pts_xy[idx[1]]
            
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist_pts = np.hypot(dx, dy)
            if dist_pts < 1e-4:
                continue
            
            # Vector normal a la recta
            nx, ny = -dy / dist_pts, dx / dist_pts
            rho = nx * p1[0] + ny * p1[1]
            
            # Distancias ortogonales
            distances = np.abs(pts_xy[:, 0] * nx + pts_xy[:, 1] * ny - rho)
            inliers = np.where(distances < self.ransac_threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_line = (nx, ny, rho)

        if best_line is None:
            rospy.logerr("[RANSAC] RANSAC fracasó encontrando el plano.")
            rospy.signal_shutdown("RANSAC failed")
            return

        nx, ny, rho = best_line
        porcentaje = (len(best_inliers) / num_pts) * 100
        rospy.loginfo(f"[RANSAC] Plano detectado: {len(best_inliers)} inliers ({porcentaje:.1f}%)")

        # ---- Búsqueda de Extremos ----
        vx, vy = -ny, nx # Vector director
        
        pts_inline = pts_xy[best_inliers]
        projections = pts_inline[:, 0] * vx + pts_inline[:, 1] * vy
        
        idx_min, idx_max = np.argmin(projections), np.argmax(projections)
        p_ext1, p_ext2 = pts_inline[idx_min], pts_inline[idx_max]

        # Profundidad mínima (Cota superior de la red)
        z_values = pts_3d[best_inliers, 2]
        min_z = float(np.min(z_values)) 
        
        # Forzar que start_net tenga el menor Norte
        if p_ext1[0] <= p_ext2[0]:
            start_xyz = [float(p_ext1[0]), float(p_ext1[1]), min_z]
            end_xyz = [float(p_ext2[0]), float(p_ext2[1]), min_z]
        else:
            start_xyz = [float(p_ext2[0]), float(p_ext2[1]), min_z]
            end_xyz = [float(p_ext1[0]), float(p_ext1[1]), min_z]

        # Redondear para evitar decimales infinitos feos en el YAML
        start_xyz = [round(v, 3) for v in start_xyz]
        end_xyz = [round(v, 3) for v in end_xyz]

        self.update_yaml(start_xyz, end_xyz)

    def update_yaml(self, start_net, end_net):
        try:
            with open(self.yaml_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            config_data['start_net'] = start_net
            config_data['end_net'] = end_net
            
            with open(self.yaml_path, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
                
            rospy.loginfo("=" * 60)
            rospy.loginfo(" ✅ YAML CONFIGURADO CON ÉXITO")
            rospy.loginfo(f"  - start_net: {start_net}")
            rospy.loginfo(f"  - end_net:   {end_net}")
            rospy.loginfo("=" * 60)
            
        except Exception as e:
            rospy.logerr(f"[RANSAC] Error escribiendo YAML: {e}")
        
        rospy.signal_shutdown("Tarea finalizada.")

if __name__ == '__main__':
    try:
        AutoPlaneEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
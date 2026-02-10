#!/usr/bin/env python3
import rospy
import numpy as np
import copy
from sensor_msgs.msg import Image
from ultralytics import YOLO

# Import new messages
from net_hole_detector.msg import BoundingBox, BoundingBoxArray

class BboxDetector:
    def __init__(self):
        rospy.init_node('bbox_detector')

        # --- 1. PARAMETROS ---
        self.model_path = rospy.get_param("~pathWeights") # Pon tu ruta por defecto
        self.conf_thres = rospy.get_param("~confidenceThreshold", 0.5)
        self.period = float(rospy.get_param("~period", 0.5))
        # self.input_topic = rospy.get_param("~input_topic", "/image_rect_color")

        # --- 2. CARGAR MODELO ---
        rospy.loginfo(f"Cargando YOLO desde: {self.model_path} ...")
        self.model = YOLO(self.model_path)
        rospy.loginfo("Modelo cargado y listo.")

        # --- 3. SUSCRIPTOR Y PUBLICADOR ---
        self.sub_img = rospy.Subscriber('camera_input', Image, self.callback_image, queue_size=1)
        self.pub_det = rospy.Publisher('yolo/detections', BoundingBoxArray, queue_size=1)

    def callback_image(self, msg):
        try:
            # 1. Imagen ROS -> Numpy (Eficiente)
            img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
            # 2. Inferencia
            # verbose=False para que no llene la consola de texto
            results = self.model(img_arr, verbose=False, conf=self.conf_thres)
            
            # 3. Preparar Mensaje de Salida
            msg_out = BoundingBoxArray()
            msg_out.header = msg.header # COPIAMOS EL TIMESTAMP ORIGINAL
            msg_out.boxes = []
            
            # 4. Rellenar datos
            result = results[0]

            if len(result.boxes) > 0:
                # xywhn devuelve: x_center, y_center, width, height (NORMALIZADOS 0-1)
                # Esto es perfecto para lo que pides.
                boxes_data = result.boxes.xywhn.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for i in range(len(boxes_data)):
                    bbox = BoundingBox()
                    bbox.class_id = str(int(classes[i]))
                    bbox.score = float(scores[i])
                    
                    # Coordenadas normalizadas (0 a 1)
                    bbox.x = float(boxes_data[i][0]) # Centro X
                    bbox.y = float(boxes_data[i][1]) # Centro Y
                    bbox.w = float(boxes_data[i][2]) # Ancho
                    bbox.h = float(boxes_data[i][3]) # Alto
                    
                    msg_out.boxes.append(bbox)
            
            # 5. Publicar
            self.pub_det.publish(msg_out)

        except Exception as e:
            rospy.logerr(f"Error YOLO: {e}")

if __name__ == '__main__':
    try:
        BboxDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

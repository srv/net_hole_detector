#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from ultralytics import YOLO, FastSAM

# Import new messages
from net_hole_detector.msg import BoundingBox, BoundingBoxArray

class BboxDetector:
    def __init__(self):
        rospy.init_node('bbox_detector')

        # --- 1. PARAMETROS ---
        self.yolo_model_path = rospy.get_param("~YOLOpathWeights") # Pon tu ruta por defecto
        self.conf_thres = rospy.get_param("~confidenceThreshold", 0.5)
        self.period = float(rospy.get_param("~period", 0.5))
        
        self.fast_sam_model_path = rospy.get_param("~FastSAMpathWeights")

        # --- 2. CARGAR MODELOS ---
        rospy.loginfo(f"Cargando YOLO desde: {self.yolo_model_path} ...")
        self.yolo_model = YOLO(self.yolo_model_path)

        rospy.loginfo(f"Cargando FastSAM desde: {self.fast_sam_model_path} ...")
        self.sam_model = FastSAM(self.fast_sam_model_path)

        rospy.loginfo("Modelos cargados y listos.")

        # --- 3. SUSCRIPTOR Y PUBLICADOR ---
        self.sub_img = rospy.Subscriber('camera_input', Image, self.callback_image, queue_size=1)
        self.pub_det = rospy.Publisher('yolo/detections', BoundingBoxArray, queue_size=1)

    def callback_image(self, msg):
        try:
            # 1. Imagen ROS -> Numpy (Eficiente)
            img_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            h_img, w_img = img_arr.shape[:2] # Guardamos dimensiones para desnormalizar

            # 2. Inferencia
            # verbose=False para que no llene la consola de texto
            results = self.yolo_model(img_arr, verbose=False, conf=self.conf_thres)
            
            # 3. Preparar Mensaje de Salida
            msg_out = BoundingBoxArray()
            msg_out.header = msg.header # COPIAMOS EL TIMESTAMP ORIGINAL
            msg_out.boxes = []
            
            # 4. Rellenar datos
            result = results[0]

            if len(result.boxes) > 0:
                # xywhn devuelve: x_center, y_center, width, height (NORMALIZADOS 0-1)
                # Esto es perfecto para lo que pides.
                # Datos de YOLO
                boxes_xywhn = result.boxes.xywhn.cpu().numpy()
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # Píxeles (Para FastSAM)
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                # --- FAST-SAM (Optimizado: Le pasamos todas las cajas a la vez) ---
                # Usamos las cajas de YOLO como 'prompts' para FastSAM
                # Esto devuelve una lista de máscaras   
                try:
                    sam_results = self.sam_model(img_arr, bboxes=boxes_xyxy, verbose=False, retina_masks=True)
                    masks = sam_results[0].masks # Puede ser None si falla
                except Exception as e:
                    rospy.logwarn(f"FastSAM error: {e}")
                    masks = None

                # Iteramos sobre cada detección
                for i in range(len(boxes_xywhn)):
                    bbox = BoundingBox()
                    bbox.class_id = str(int(classes[i]))
                    bbox.score = float(scores[i])
                    
                    # --- A. Datos YOLO (La caja) ---
                    bbox.x = float(boxes_xywhn[i][0])
                    bbox.y = float(boxes_xywhn[i][1])
                    bbox.w = float(boxes_xywhn[i][2])
                    bbox.h = float(boxes_xywhn[i][3])
                    
                    # --- B. Datos FastSAM (La máscara) ---
                    bbox.has_mask = False
                    bbox.mask_area_px = 0.0
                    bbox.mask_x_norm = bbox.x # Fallback: Usamos centro caja
                    bbox.mask_y_norm = bbox.y # Fallback: Usamos centro caja

                    if masks is not None and len(masks) > i:
                        try:
                            # Sacar la máscara 'i'
                            mask_data = masks.data[i].cpu().numpy() # Matriz 0 y 1
                            
                            # 1. Área en Píxeles
                            area_px = np.count_nonzero(mask_data)
                            bbox.mask_area_px = float(area_px)

                            # 2. Centroide Real (Momentos)
                            M = cv2.moments(mask_data)
                            if M["m00"] != 0:
                                cX = M["m10"] / M["m00"]
                                cY = M["m01"] / M["m00"]
                                # Normalizamos para enviar (0-1)
                                bbox.mask_x_norm = float(cX / w_img)
                                bbox.mask_y_norm = float(cY / h_img)
                                bbox.has_mask = True
                        except Exception as inner_e:
                            pass # Si falla una máscara, seguimos con las otras


                    msg_out.boxes.append(bbox)
            
            # 5. Publicar
            self.pub_det.publish(msg_out)

        except Exception as e:
            rospy.logerr(f"Error BboxDetector: {e}")

if __name__ == '__main__':
    try:
        BboxDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

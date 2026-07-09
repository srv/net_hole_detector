#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from ultralytics import YOLO


# Import new messages
from net_hole_detector.msg import BoundingBox, BoundingBoxArray

# Import auxiliary classes
from net_hole_detector.scale_estimator import ScaleEstimator
from net_hole_detector.ros_numpy_converter import RosNumPyConverter

CONF_THRESHOLD = 0.40
DARK_PERCENTILE = 80
ALPHA = 0.55

class BboxDetector:
    def __init__(self):
        rospy.init_node('bbox_detector')

        # --- 1. PARAMETROS ---
        self.model_path = rospy.get_param("~pathWeights") # Pon tu ruta por defecto
        self.conf_thres = rospy.get_param("~confidenceThreshold", 0.5)
        self.period = float(rospy.get_param("~period", 0.5))
        self.is_lab_method = rospy.get_param("~is_lab_method", True)
        # Variable to store last time we process an image
        self.last_process_time = rospy.Time(0)
        # self.input_topic = rospy.get_param("~input_topic", "/image_rect_color")

        # --- 2. CARGAR MODELO ---
        rospy.loginfo(f"Cargando YOLO desde: {self.model_path} ...")
        self.model = YOLO(self.model_path)
        rospy.loginfo("Modelo cargado y listo.")

        # --- 3. CARGAR CLASES ---
        self.scale_estimator = ScaleEstimator()
        self.bridge = RosNumPyConverter()

        # --- 4. SUSCRIPTOR Y PUBLICADOR ---
        self.sub_img = rospy.Subscriber('camera_input', Image, self.callback_image, queue_size=1)
        self.pub_det = rospy.Publisher('yolo/detections', BoundingBoxArray, queue_size=1)
        self.pub_debug = rospy.Publisher('yolo/debugging', Image, queue_size=1)

    def callback_image(self, msg):
        # THROTTLE
        now = rospy.Time.now()

        # Calculate time since last processed image
        if (now - self.last_process_time).to_sec() < self.period:
            # if less time passed we ignore the image
            return
        
        # else, we update the clock
        self.last_process_time = now
        
        try:
            # 1. Imagen ROS -> Numpy (Using auxiliary class bridge)
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            output_rgb = cv_img.copy()
            
            # 2. Inferencia
            # verbose=False para que no llene la consola de texto
            results = self.model(cv_img, verbose=False, conf=self.conf_thres)
            
            # 3. Preparar Mensaje de Salida
            msg_out = BoundingBoxArray()
            msg_out.header = msg.header # COPIAMOS EL TIMESTAMP ORIGINAL
            msg_out.boxes = []
            
            # 4. Process detections
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
                                
                    cls = int(result.boxes[i].cls[0])
                    x1, y1, x2, y2 = result.boxes[i].xyxy[0].tolist()

                    x1 = max(0, min(int(round(x1)), msg.width - 1))
                    y1 = max(0, min(int(round(y1)), msg.height - 1))
                    x2 = max(0, min(int(round(x2)), msg.width - 1))
                    y2 = max(0, min(int(round(y2)), msg.height - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Crop dins la bounding box
                    crop = cv_img[y1:y2, x1:x2, :].astype(np.float32)
                    if self.is_lab_method:
                        lab = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2LAB)
                        L = lab[:,:,0]
                        darkness = 255 - L
                    else:
                        R = crop[:, :, 0]
                        G = crop[:, :, 1]
                        B = crop[:, :, 2]

                        # Luminància percebuda
                        luminance = 0.299 * R + 0.587 * G + 0.114 * B

                        # Foscor: valor alt = píxel més fosc
                        darkness = 255.0 - luminance

                    dark_threshold = np.percentile(darkness, DARK_PERCENTILE)
                    dark_mask = darkness >= dark_threshold

                    # Pintar píxels foscos
                    crop_out = output_rgb[y1:y2, x1:x2, :]

                    dark_overlay = crop_out.copy()
                    dark_overlay[dark_mask] = [0, 255, 255]  # cian

                    crop_out[dark_mask] = (
                        (1 - ALPHA) * crop_out[dark_mask]
                        + ALPHA * dark_overlay[dark_mask]
                    ).astype(np.uint8)

                    output_rgb[y1:y2, x1:x2, :] = crop_out

                    # -------------------------------
                    # Centre i desviació estàndard
                    # -------------------------------
                    ys, xs = np.where(dark_mask)

                    if len(xs) > 0:

                        # Coordenades locals dins la BB
                        mean_x_local = np.mean(xs)
                        mean_y_local = np.mean(ys)

                        std_x = np.std(xs)
                        std_y = np.std(ys)
                        matriu_cov = np.cov(np.vstack([xs, ys]))
                        det_cov = np.linalg.det(matriu_cov)

                        # Coordenades globals dins la imatge
                        mean_x_global = int(round(x1 + mean_x_local)) # TODO: aquestes són les coordenades del punt central de la corrosio. A incloure en el missatge del topic
                        mean_y_global = int(round(y1 + mean_y_local))

                        bbox.corr_x = mean_x_global / msg.width
                        bbox.corr_y = mean_y_global / msg.height
                        bbox.std_x = std_x
                        bbox.std_y = std_y
                        bbox.area = 4 * np.pi * np.sqrt(det_cov) if det_cov > 0 else 0.0
                        bbox.width = msg.width
                        bbox.height = msg.height

                        # Pintar centre
                        cv2.circle(
                            output_rgb,
                            (mean_x_global, mean_y_global),
                            radius=5,
                            color=(255, 0, 255),  # magenta
                            thickness=-1
                        )

                        # Pintar el·lipse de desviació estàndard
                        cv2.ellipse(
                            output_rgb,
                            center=(mean_x_global, mean_y_global),
                            axes=(int(round(std_x)), int(round(std_y))),
                            angle=0,
                            startAngle=0,
                            endAngle=360,
                            color=(255, 0, 255),  # magenta
                            thickness=2
                        )

                        # Opcional: pintar 2 sigma amb línia més fina
                        cv2.ellipse(
                            output_rgb,
                            center=(mean_x_global, mean_y_global),
                            axes=(int(round(2 * std_x)), int(round(2 * std_y))),
                            angle=0,
                            startAngle=0,
                            endAngle=360,
                            color=(255, 150, 255),
                            thickness=1
                        )

                    # Dibuixar bounding box
                    cv2.rectangle(
                        output_rgb,
                        (x1, y1),
                        (x2, y2),
                        color=(0, 255, 0),
                        thickness=2
                    )

                    label = f"corrosion {bbox.score:.2f}"

                    cv2.putText(
                        output_rgb,
                        label,
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

                msg_out.boxes.append(bbox)

                # 6. Publicar
                self.pub_det.publish(msg_out)
                self.pub_debug.publish(self.bridge.cv2_to_imgmsg(output_rgb))

        except Exception as e:
            rospy.logerr(f"Error YOLO: {e}")

if __name__ == '__main__':
    try:
        BboxDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from os import getenv
from statistics import mean
import torch
from torchvision.transforms.functional import to_tensor, resize, normalize
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle

# Import new messages
from net_hole_detector.msg import BoundingBox, BoundingBoxArray, NetStats

# Import auxiliary classes
from net_hole_detector.scale_estimator import ScaleEstimator
from net_hole_detector.ros_numpy_converter import RosNumPyConverter

MODEL_NAME = "dinov3_vitl16"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_TO_NUM_LAYERS = 24
PATCH_SIZE = 16
IMAGE_SIZE = 768 # TODO: actualitzar amb tamany que toca
DINOV3_LOCATION = getenv("DINOV3_LOCATION")
DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

class BboxDetector:
    def __init__(self):
        rospy.init_node('bbox_detector')

        # --- 1. PARAMETROS ---
        self.model_path = rospy.get_param("~pathWeights") # Pon tu ruta por defecto
        self.period = float(rospy.get_param("~period", 0.5))
        # Variable to store last time we process an image
        self.last_process_time = rospy.Time(0)
        # self.input_topic = rospy.get_param("~input_topic", "/image_rect_color")

        # --- 2. CARGAR MODELO ---
        rospy.loginfo(f"Cargando YOLO desde: {self.model_path} ...")
        self.model = torch.hub.load(repo_or_dir=DINOV3_GITHUB_LOCATION, model=MODEL_NAME, source="github", weights="https://urldefense.com/v3/__https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNG5wN3ZtMDcyYmFidGlvZzAyeTZzMGJiIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3Nzg0MTI0NDJ9fX1dfQ__&Signature=ZdOD*7EeKXtqPBNsOJUDllCLOfu12rCD3tQPkmB15obXoi3nYO5-0wAzzw2Dt0ikFFD*7EMbAGmZ6VoasKPqgwM5cy0AixgeqCyaNVtQsSCq-Gelx6fEbE6s41n5Vae1wq0zj4eMXKH2Shinf39UimxqWqARmAxPp*7EE1AVVuIFTWP1kKx4JYOoQMw-eRgYRDX1OXxO8ZJTTxEf0mA3FCnQFS*7EhByG8yMkNCsnRRGogn8d0J19ebUcXyX*7Ev0aV7fUJm-2YYl0VsrBAVwducA4SYs4n4eIRGdsKmLHwDQf-u-T5iO1lV8MJddkjazr2rwSRm5E*7E625dTC3ACbqoQVwGC7C8A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1477722903817465__;JSUlJSUl!!D9dNQwwGXtA!UXvVTsNq3Zgg52jtEN8Nwmpatts3_jB7o7zwtsAO4nmFZ-3b9netdFOaqgwNS4xOHjEqt2TWj9GCSq0OuCXoCg$")
        self.model.cuda()
        self.model.eval()
        rospy.loginfo("Modelo cargado y listo.")

        # --- 3. CARGAR CLASES ---
        self.bridge = RosNumPyConverter()

        # --- 4. CARGAR REF IMAGES ---
        # Codi per carregar els centroides dels clusters i els features de les imatges de referència
        self.__ref_image_feats = None
        self.__clustering_centroids_list = list()

        # --- 5. SUSCRIPTOR Y PUBLICADOR ---
        self.sub_img = rospy.Subscriber('camera_input', Image, self.callback_image, queue_size=1)
        self.pub_det = rospy.Publisher('yolo/detections', BoundingBoxArray, queue_size=1)

        # --- 6. NEW BLOBS PUBLISHERS ---
        self.pub_blob_bin = rospy.Publisher('/net_hole_detector/blobs/binary', Image, queue_size=1)
        # self.pub_blob_overlay = rospy.Publisher('/net_hole_detector/blobs/overlay', Image, queue_size=1)

    def callback_image(self, msg):
        # THROTTLE
        now = rospy.Time.now()

        # Calculate time since last processed image
        if (now - self.last_process_time).to_sec() < self.period:
            # if less time passed we ignore the image
            return
        
        # else, we update the clock
        self.last_process_time = now
        
        # try:
        #     # 1. Imagen ROS -> Numpy (Using auxiliary class bridge)
        #     cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
        #     # 2. Inferencia
        #     # verbose=False para que no llene la consola de texto
        #     img_resized = to_tensor(resize(cv_img, (30 * 16, 30 * 16)))
        #     img_resized_norm = normalize(img_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        #     img_resized_norm = img_resized_norm.unsqueeze(0).cuda()
            
        #     outputs = self.model.forward_features(img_resized_norm)
        #     image_feats = outputs["x_norm_clstoken"]   # shape: (B, D)
        #     image_feats = image_feats.squeeze(0)
        #     image_feats = np.asarray(image_feats.cpu())

        #     # Get patch features
        #     feats = self.model.get_intermediate_layers(img_resized_norm, n=range(MODEL_TO_NUM_LAYERS), reshape=True, norm=True)
        #     x = feats[-1].squeeze().detach().cpu() # Agafem els features de la darrera capa. Totes les altres són el output de les anteriors.
        #     shaping = x.shape
        #     dim = shaping[0]
            
        #     x = x.view(dim, -1).permute(1, 0)
        #     patch_features = torch.nn.functional.normalize(x, p=2, dim=0)
        #     patch_features_array = np.asarray(patch_features)

        #     # 3. Compute locations # TODO: Això es podria fer una vegada al principi i deixar-ho guardat. Les imatges sempre tindran la mateixa mida
        #     # compute the corresponding 2D patch locations in the anomalies image
        #     n_patches = shaping[1] * shaping[2]
        #     patch_indices = torch.arange(n_patches)
        #     locs_2d = (
        #         torch.stack(
        #             (
        #                 patch_indices // shaping[2],  # row
        #                 patch_indices % shaping[2]    # column
        #             ),
        #             dim=-1
        #         ) + 0.5
        #     ) * PATCH_SIZE

        #     index = self.__normal_image_selection(self.__ref_image_feats, image_feats[0])

        #     ### Calcular patch similarity with selected reference Image
        #     num_row, num_col = np.shape(patch_features_array)
        #     patch_similarity_average = np.zeros((num_row, 1))
        #     for l, patch in enumerate(patch_features_array):
        #         features = np.asarray(patch)
        #         patch_similarity_list = []
        #         for cluster_centroid in self.__clustering_centroids_list[index]:
        #             patch_similarity = np.dot(features, cluster_centroid) / (np.linalg.norm(features) * np.linalg.norm(cluster_centroid))
        #             patch_similarity_list.append(patch_similarity)
        #         patch_similarity_average[l, 0] = mean(patch_similarity_list)


        #     corr_im = np.asarray(np.transpose(img_resized, (1, 2, 0)))
        #     red_channel = (corr_im[:,:,0] * 255).astype(np.uint8)
        #     median_red = np.quantile(red_channel, 0.75)
        #     selected_elements = (red_channel>median_red) | (((corr_im[:,:,0]<100) & (corr_im[:,:,0]>20)) &  ((corr_im[:,:,1]<100) & (corr_im[:,:,1]>20)) & ((corr_im[:,:,2]<100) & (corr_im[:,:,2]>20)))
            
        #     scaler = MinMaxScaler()
        #     normalized_data = 1 - scaler.fit_transform(patch_similarity_average)
        #     new_im_heatmap = np.zeros(shape=(img_resized.shape[1], img_resized.shape[2]), dtype=np.int16)
        #     for j in locs_2d:
        #         row_pos = int(locs_2d[j, 0].item())
        #         col_pos = int(locs_2d[j, 1].item())
        #         new_im_heatmap[row_pos-8:row_pos+7, col_pos-8:col_pos+7] = normalized_data[j] * 255

        #     threshold = max(np.quantile(new_im_heatmap, 0.875), 75)
        #     segmented = np.where(new_im_heatmap<threshold, 0, 255)
        #     resultat  = np.where(selected_elements, segmented, 0)

        #     kernel = np.ones((3,3),np.uint8)
        #     resultat = cv2.morphologyEx(resultat.astype('uint8'), cv2.MORPH_OPEN, kernel)
        #     resultat = cv2.morphologyEx(resultat.astype('uint8'), cv2.MORPH_OPEN, kernel)
        #     kernel = np.ones((5,5),np.uint8)
        #     resultat = cv2.morphologyEx(resultat, cv2.MORPH_CLOSE, kernel)

        #     self.pub_blob_bin.publish(self.bridge.cv2_to_imgmsg(resultat, "mono8"))

        #     contours, hierarchi = cv2.findContours(resultat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #     largest_contour = max(contours, key=cv2.contourArea)

        #     # bounding box
        #     x, y, w, h = cv2.boundingRect(largest_contour)
        #     print('This is w = ' + str(w) + '. And h = ' + str(h))

        #     # 4. Preparar Mensaje de Salida
        #     msg_out = BoundingBoxArray()
        #     msg_out.header = msg.header # COPIAMOS EL TIMESTAMP ORIGINAL
        #     msg_out.boxes = []
            
        #     # 5. Rellenar datos

        #     if contours is not None:
        #         # xywhn devuelve: x_center, y_center, width, height (NORMALIZADOS 0-1)
        #         # Esto es perfecto para lo que pides.

        #         bbox = BoundingBox()
        #         bbox.class_id = 0
        #         bbox.score = 0
                
        #         # Coordenadas normalizadas (0 a 1)
        #         bbox.x = float(x) # Centro X
        #         bbox.y = float(y) # Centro Y
        #         bbox.w = float(w) # Ancho
        #         bbox.h = float(h) # Alto
                
        #         msg_out.boxes.append(bbox)
            
        #     # 6. Publicar
        #     self.pub_det.publish(msg_out)

        # except Exception as e:
        #     rospy.logerr(f"Error YOLO: {e}")

    def __normal_image_selection(image_features_list_normal, image_features_corr):

        max_similarity = 0
        index = 0
        for i, image_features in enumerate(image_features_list_normal):

            image_similarity = np.dot(image_features_corr, image_features) / (np.linalg.norm(image_features_corr) * np.linalg.norm(image_features))
            print('Patch similarity = ' + str(image_similarity))

            if max_similarity < image_similarity:
                max_similarity = image_similarity
                index = i

        return index

if __name__ == '__main__':
    try:
        BboxDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

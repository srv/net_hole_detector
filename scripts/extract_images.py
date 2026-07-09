#!/usr/bin/env python3
import rospy
import cv2
import os
import subprocess
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class ImageSaver:
    def __init__(self, topic_name, output_dir):
        self.bridge = CvBridge()
        self.topic_name = topic_name
        self.output_dir = output_dir
        self.count = 0
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            rospy.loginfo(f"Directori creat: {self.output_dir}")

        # Subscripció al tòpic d'imatges
        self.image_sub = rospy.Subscriber(self.topic_name, CompressedImage, self.callback)
        rospy.loginfo(f"Escoltant el tòpic: {self.topic_name}")

    def callback(self, data):
        try:
            # Convertir el missatge de ROS a format OpenCV (bgr8 és l'estàndard)
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
            
            # Generar el nom del fitxer (pots usar el timestamp si ho prefereixes)
            filename = os.path.join(self.output_dir, f"frame_{self.count:04d}.png")
            # cv2.imwrite(filename, cv_image)
            
            rospy.loginfo(f"Guardat: {filename}")
            self.count += 1
        except CvBridgeError as e:
            rospy.logerr(f"Error en CvBridge: {e}")

def run_bag(bag_path, rate):
    # Comanda per reproduir el bag a la freqüència desitjada
    command = ["rosbag", "play", bag_path, "-r", str(rate)]
    rospy.loginfo(f"Executant: {' '.join(command)}")
    return subprocess.Popen(command)

if __name__ == '__main__':
    rospy.init_node('image_extractor_node', anonymous=True)

    # CONFIGURACIÓ
    BAG_FILE = '/home/rosuser/repo/catkin_ws/src/net_hole_detector/atlantis/girona1000_atlantis_2022-11-17-16-32-33_0.bag'
    TOPIC_NAME = '/girona1000/flir_spinnaker_camera/image_raw/compressed'
    OUTPUT_FOLDER = '/home/rosuser/repo/catkin_ws/src/net_hole_detector/atlantis/images/'
    RATE = 25 # Freqüència de reproducció (4x)   /home/rosuser/repo/catkin_ws/src/net_hole_detector/atlantis/girona1000_intervention_2022-07-20-13-05-55_0.bag

    # Iniciar el guardador d'imatges
    saver = ImageSaver(TOPIC_NAME, OUTPUT_FOLDER)

    # Iniciar la reproducció del bag
    bag_process = run_bag(BAG_FILE, RATE)

    # Mantenir el node viu fins que el bag acabi o matem el procés
    while not rospy.is_shutdown():
        if bag_process.poll() is not None:  # Si el procés del bag ha acabat
            rospy.loginfo("El rosbag ha finalitzat.")
            break
        rospy.sleep(1)

    rospy.signal_shutdown("Feina feta")
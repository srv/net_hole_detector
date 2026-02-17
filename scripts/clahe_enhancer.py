#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ClaheEnhancer:
    def __init__(self):
        rospy.init_node('clahe_enhancer')
        
        # Parametros
        self.clip_limit = rospy.get_param("~clip_limit", 2.0)
        self.grid_size = int(rospy.get_param("~grid_size", 8))
        
        # Objetos
        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.grid_size, self.grid_size))
        
        # Suscriptor (Topic de entrada se define en el launch)
        self.sub = rospy.Subscriber("image_in", Image, self.callback)
        
        # Publicador (Topic de salida se define en el launch)
        self.pub = rospy.Publisher("image_out", Image, queue_size=1)
        
        rospy.loginfo(f"CLAHE Node Started. Clip: {self.clip_limit}, Grid: {self.grid_size}")

    def callback(self, msg):
        try:
            # 1. ROS -> OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # 2. Aplicar CLAHE (Lab channel L)
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. OpenCV -> ROS
            out_msg = self.bridge.cv2_to_imgmsg(enhanced_img, encoding="bgr8")
            out_msg.header = msg.header # Importante mantener el timestamp original
            
            self.pub.publish(out_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CLAHE Error: {e}")

if __name__ == '__main__':
    ClaheEnhancer()
    rospy.spin()
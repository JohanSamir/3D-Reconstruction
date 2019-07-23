#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
import sys
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
import open3d as opn3
import subprocess
import os
from pandas import DataFrame
import pandas as pd
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import FusionData

class DepthStimationMain():

	def __init__(self):

		self.pc_pub = rospy.Publisher('/Dense',PointCloud2,queue_size=2)
		self.image_pub = rospy.Publisher("/image_output",Image, queue_size = 2)
		self.a = 0

	def depthSt(self, file_list, ns):
		DepthMap = FusionData.depth_laser_camer(file_list,self.a,ns)
		plt.figure(figsize=(14,10),frameon=False)
		plt.imshow(np.asarray(DepthMap))
		plt.axis('off')
		plt.savefig('/home/johan/Documents/Alignment/DatasetKitti/DepthMaps/'+str(a)+'.png',bbox_inches='tight',pad_inches=0)
		print('Finish')
		#self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

	def ICP(self):
		print('a')
		return M

def main(args):
	print('Starting...')
	rospy.init_node('sync_node', anonymous=True)
	rospy.loginfo("sync_node on")

	# Find Images
	point_dir = '/home/johan/Documents/Alignment/DatasetKitti/Images/'
	file_list = []
	file_glob = os.path.join(point_dir, '*.png')
	file_list.extend(gfile.Glob(file_glob))
	file_list = np.sort(file_list)

	# Find Pointclouds
	point_dir = '/home/johan/Documents/Alignment/DatasetKitti/PCD/'
	file_list_pcd = []
	file_glob = os.path.join(point_dir, '*.pcd')
	file_list_pcd.extend(gfile.Glob(file_glob))
	file_list_pcd = np.sort(file_list_pcd)

	#segments => ns (5000 OK)
	ns = 1000 
	sc = DepthStimationMain()
	sc.depthSt(file_list,file_list_pcd,ns)
  
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
  
    main(sys.argv)


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
		self.bridge = CvBridge()

	def depthSt(self, file_list, ns):

		for point_dir in file_list:

			imageMsg=Image
			DepthMap = FusionData.depth_laser_camer(point_dir,self.a,ns)
			print(DepthMap.shape, type(DepthMap))

			img = np.stack((DepthMap,) * 3,-1) 
			img = img.astype(np.uint8) 
			grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			#imageMsg.height = DepthMap.shape[0]
			#imageMsg.width = DepthMap.shape[0]
			#imageMsg.data = DepthMap
			
			plt.figure(figsize=(14,10),frameon=False)
			plt.imshow(np.asarray(DepthMap))
			plt.axis('off')
			plt.savefig('/home/johan/Documents/Alignment/DatasetKitti/DepthMaps/'+str(self.a)+'.png',bbox_inches='tight',pad_inches=0)
			print('Finish')
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(grayed, "mono8"))
			#self.image_pub.publish(imageMsg)
			self.a = self.a +1

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
	sc.depthSt(file_list,ns)
  
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
  
    main(sys.argv)


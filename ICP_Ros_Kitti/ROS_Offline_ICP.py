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


class synchronizer():

	def __init__(self):

		self.pc_pub = rospy.Publisher('/Dense',PointCloud2,queue_size=2)
		self.pc_pub2 = rospy.Publisher('/Dense_ICP',PointCloud2,queue_size=2)
		self.a = 0
		self.PCdense = np.array([])
		self.ms = PointCloud2()
		self.mss = PointCloud2()
	
	def selec(self, file_list):
		# There are approximately 154 pointclouds
		for point_dir in file_list[0:len(file_list):2]:
			if self.a == 0:
				refe = pd.read_csv(point_dir)
				df = DataFrame(refe, columns= ['x', 'y','z'])
				pointcloud1_path_refe = "/home/johan/Desktop/UAO Projects_2 final/KitiiDatabase/PointCloud_CSV_kitti_1/refe_kitti.csv"
				export_csv = df.to_csv (pointcloud1_path_refe, index = None, header=True) #Don't forget to add '.csv' at the end of the path
				self.PCdense = np.array(list(df))
				self.ms.data = self.PCdense
				self.pc_pub.publish(self.ms)
				self.mss.data = self.PCdense
				self.pc_pub2.publish(self.mss)
				print(self.a)

			else:
				ref = pd.read_csv(point_dir)
				df = np.array(DataFrame(refe, columns= ['x', 'y','z']))
				#print('df',df.shape)
				self.ms.data = df
				self.pc_pub.publish(self.ms)

				self.ICP(pointcloud1_path_refe,point_dir)
				Out_ICP = pd.read_csv('/home/johan/refe_OutKK.csv')
				Out_ICP = np.array(DataFrame(Out_ICP, columns= ['x', 'y','z']))
				Out_ICP = Out_ICP[1:Out_ICP.shape[0],:]
				f = np.vstack((self.PCdense,Out_ICP))
				#print('f',f.shape) 				
				self.PCdense = list(f)
				fsave = DataFrame(f)
				fsave.to_csv (pointcloud1_path_refe, index = None, header=True) #Don't forget to add '.csv' at the end of the path
				self.mss.data = self.PCdense
				self.pc_pub2.publish(self.mss)

			self.a = self.a + 1
			print(self.a)

	def ICP(self,pointcloud1,pointcloud2):
		#--------------- correr el ejecutable de c++ ------------------
		icp_path="/home/johan/Libraries/libpointmatcher/build/examples"
		runicp = subprocess.Popen([os.path.join(icp_path,"icp_simple"),pointcloud1,pointcloud2,'OutKK'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		runicp.wait()
    
def main(args):
	print('Starting...')
	rospy.init_node('sync_node', anonymous=True)
	rospy.loginfo("sync_node on")

	# Find number of PointClouds
	point_dir = '/home/johan/Desktop/UAO Projects_2 final/KitiiDatabase/PointCloud_CSV_kitti_1/'
	file_list = []
	file_glob = os.path.join(point_dir, '*.csv')
	file_list.extend(gfile.Glob(file_glob))
	file_list = np.sort(file_list)
	#print(len(file_list))
	#print(type(file_list[0]),file_list[0:8])
	sc = synchronizer()
	sc.selec(file_list)
  
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
  
    main(sys.argv)
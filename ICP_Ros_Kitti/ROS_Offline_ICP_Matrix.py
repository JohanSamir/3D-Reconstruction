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
		#for point_dir in file_list[0:len(file_list):2]:
		for point_dir in file_list:
			if self.a == 0:
				refe = pd.read_csv(point_dir)
				df = DataFrame(refe,columns= ['x','y','z'])
				#print(df.shape)
				pointcloud1_path_refe = "/home/johan/Documents/Alignment/INFo_Bag8/PCD_Transformadas/0.csv"
				export_csv = df.to_csv (pointcloud1_path_refe, index = None, header=True) #Don't forget to add '.csv' at the end of the path
				self.PCdense = np.array(list(df))
				self.ms.data = self.PCdense
				self.pc_pub.publish(self.ms)
				self.mss.data = self.PCdense
				self.pc_pub2.publish(self.mss)
				MAtrixAcu = np.ones((4,4))
				#print(self.a)

			else:
				print(self.a)
				ref = pd.read_csv(point_dir)
				df = np.array(DataFrame(ref,columns= ['x','y','z']))
				#print('df',df.shape,df[0])
				colum = np.ones((df.shape[0],1))
				df_colum = np.hstack((df,colum))
				#print('df_colum',df_colum.shape,df_colum[0], type(df_colum))
				self.ms.data = df
				self.pc_pub.publish(self.ms)

				Matrix = self.ICP(pointcloud1_path_refe,point_dir)
				#print('Matrix_ICP',Matrix)
				# It is important to set np.dot when it is a premultiplication not * (4X4)
				MAtrixAcu = np.multiply(Matrix,MAtrixAcu)
				#print('MAtrixAcu',MAtrixAcu)
				PointMul = np.dot(df_colum,MAtrixAcu)
				#print(PointMul[0])
				PointMul = PointMul[:,0:3]

				fsave = DataFrame(PointMul,columns= ['x','y','z'])
				fsave.to_csv ('/home/johan/Documents/Alignment/INFo_Bag8/PCD_Transformadas/'+str(self.a)+'.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
				self.mss.data = self.PCdense
				self.pc_pub2.publish(self.mss)
				pointcloud1_path_refe = point_dir

			self.a = self.a + 1
			

	def ICP(self,pointcloud1,pointcloud2):
		#--------------- correr el ejecutable de c++ ------------------
		icp_path="/home/johan/Libraries/libpointmatcher/build/examples"
		runicp = subprocess.Popen([os.path.join(icp_path,"icp_simple"),pointcloud1,pointcloud2,'OutKK'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		runicp.wait()

		stdout, stderr = runicp.communicate()
		# bytes to str, alternative
		M = np.matrix(str((stdout.decode("utf-8"))))
		M = M.reshape(4,4)
		#print(M, M.shape)
		return M

def main(args):
	print('Starting...')
	rospy.init_node('sync_node', anonymous=True)
	rospy.loginfo("sync_node on")

	# Find number of PointClouds
	#point_dir = '/home/johan/Desktop/UAO Projects_2 final/KitiiDatabase/PointCloud_CSVKitti_xyzr/'
	#point_dir = '/home/johan/Documents/Alignment/Points_csv/'
	point_dir = '//home/johan/Documents/Alignment/INFo_Bag8/Points_csv/'
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
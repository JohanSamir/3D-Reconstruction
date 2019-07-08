
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

class synchronizer():

	def __init__(self,file_list):

		self.pc_pub = rospy.Publisher('/Dense',PointCloud2,queue_size=2)
		self.pc_pub2 = rospy.Publisher('/Dense_ICP',PointCloud2,queue_size=2)
		self.a = 0
		self.PCdense = np.array([])
		self.ms = PointCloud2()
		self.mss = PointCloud2()
	
	def selec(self, file_list):

		for point_dir in file_list:
			if a == 0:
				refe = pd.read_csv(point_dir)
				df = DataFrame(refe, columns= ['x', 'y','z'])
				pointcloud1_path_refe = "/home/johan/Documents/Alignment/Points_csv/refe.csv"
				export_csv = df.to_csv (pointcloud1_path_refe, index = None, header=True) #Don't forget to add '.csv' at the end of the path
				self.PCdense = np.array(df)
				self.ms.data = self.PCdense
				self.pc_pub.publish(self.ms)
				self.mss.data = self.PCdense
				self.pc_pub2.publish(self.mss)
				print(a)

			else:
				ref = pd.read_csv(point_dir)
				df = np.array(DataFrame(refe, columns= ['x', 'y','z']))
				self.ms.data = df
				self.pc_pub.publish(self.ms)

				ICP(self,pointcloud1_path_refe,point_dir)
				Out_ICP = pd.read_csv('/home/johan/refe_OutKK.csv')
				Out_ICP = np.array(DataFrame(Out_ICP, columns= ['x', 'y','z']))
				Out_ICP = Out_ICP[1:Out_ICP.shape[0],:]

				self.PCdense = np.vstack((self.PCdense,Out_ICP)) 
				self.mss.data = self.PCdense
				self.pc_pub2.publish(self.mss)
				print(self.a)

	def ICP(self,pointcloud1,pointcloud2):
		#--------------- correr el ejecutable de c++ ------------------
		icp_path="/home/johan/Libraries/libpointmatcher/build/examples"
		runicp = subprocess.Popen([os.path.join(icp_path,"icp_simple"),pointcloud1,addres2,'OutKK'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		runicp.wait()
    
def main(args):
	print('Starting...')
	rospy.init_node('sync_node', anonymous=True)
	rospy.loginfo("sync_node on")

	# Find number of PointClouds
	point_dir = '/home/johan/Documents/Alignment/Points_csv/'
	#se crea un vector vacio para guardar los nombres de las imágenes.
	file_list = []
	#se obtienen todos los nombre de las imagenes en la carpeta train que tengan extensión .csv
	file_glob = os.path.join(point_dir, '*.csv')
	#se organizan en el vector file_list todos las direcciones de las imágenes 
	#encontrados dentro de la carpeta train
	file_list.extend(gfile.Glob(file_glob))
	file_list = np.sort(file_list)
	#print(len(file_list))
	#print(type(file_list[0]),file_list[0:8])
	sc = synchronizer(file_list)
	sc.
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  
    main(sys.argv)
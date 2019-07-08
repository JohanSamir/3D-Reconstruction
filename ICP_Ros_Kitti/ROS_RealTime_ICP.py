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



class synchronizer:

  def __init__(self):
    #self.laserProj = LaserProjection()
    #self.bridge = CvBridge()
    #self.image_sub = message_filters.Subscriber('/left/image_rect_color', Image)
    self.pc_pub = rospy.Publisher('/Dense',PointCloud2,queue_size=2)
    self.pc_sub = rospy.Subscriber('/kitti/velo/pointcloud', PointCloud2, self.callback)

    #self.pc_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
    #self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pc_sub], 10, 0.1, allow_headerless=False)
    #self.ts.registerCallback(self.callback)
    self.matref = np.ones((4,4))
    self.a = 0
    self.PCdense = np.array([])
    self.ms = PointCloud2()

  def callback(self, pointcloud):

	lidarPC2 = pc2.read_points(pointcloud)
	lidar = np.array(list(lidarPC2))
	#lidarPointsXYZ = list(lidar[:,0:3])
	lidarPointsXYZ = lidar[:,0:3]


	if self.a ==0:
		self.PCdense = lidarPointsXYZ
		print(lidarPointsXYZ.shape, type(lidarPointsXYZ))
		#print('lidarPointsXYZ',type(lidarPointsXYZ), lidarPointsXYZ[0])
		d1 = DataFrame(lidarPointsXYZ)
		self.addres = '/home/johan/Documents/Alignment/Points_csv/refe1.csv'
		d1.to_csv (self.addres, index = None, header=True) #Don't forget to add '.csv' at the end of the path
		self.ms.data = self.PCdense
		self.pc_pub.publish(self.ms)
		print(self.a)
	else:
		#matrix_ICP = self.ICP(self.PCdense,lidarPointsXYZ)
		matrix_ICP = self.ICP(self.addres,lidarPointsXYZ)
		#print('matrix_ICP',matrix_ICP,'\n','lidarPointsXYZ',lidarPointsXYZ.shape
		#refe = pd.read_csv('/home/johan/repos/GitHub/3D-Reconstruction/refe_OutPc.csv')
		refe = pd.read_csv('/home/johan/refe_OutKK.csv')
		refe = np.array(refe)
		print(refe.shape,type(refe))
		#c = list(np.vstack((self.PCdense,refe)))
		c = np.vstack((self.PCdense,refe))
		print(c.shape,type(c))
		#print('c',type(c),c[0])
		self.PCdense = c
		self.ms.data = self.PCdense
		self.pc_pub.publish(self.ms)
		print('HI')
	self.a = self.a+1
	print(self.a)


  def ICP(self,pointcloud1,pointcloud2):
	#--------------- correr el ejecutable de c++ ------------------
	d2 = DataFrame(pointcloud2)
	addres2 = '/home/johan/Documents/Alignment/Points_csv/refe2.csv'
	export_csv = d2.to_csv (addres2, index = None, header=True) #Don't forget to add '.csv' at the end of the path
	icp_path="/home/johan/Libraries/libpointmatcher/build/examples"
	runicp = subprocess.Popen([os.path.join(icp_path,"icp_simple"),pointcloud1,addres2,'OutKK'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	runicp.wait()
	#stdout, stderr = runicp.communicate()
	#print(stdout,'\n') # bytes to str, alternative
	#sh = np.matrix(str(stdout, encoding='ascii'))
	#sh = sh.reshape(4,4)
	#print(sh, sh.shape)

	#return sh
    
def main(args):
  print('Starting...')
  rospy.init_node('sync_node', anonymous=True)
  rospy.loginfo("sync_node on")
  a = 0
  sc = synchronizer()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  
    main(sys.argv)
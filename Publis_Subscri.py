#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
import sys
from cv_bridge import CvBridge, CvBridgeError
#import cv2
import numpy as np
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
import open3d as opn3

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

  def callback(self, pointcloud):
    print('callback')
    lidarPC2 = pc2.read_points(pointcloud)
    lidar = np.array(list(lidarPC2))
    lidarPointsXYZ = lidar[:,0:3]
    self.pc_pub.publish(pointcloud)
    
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


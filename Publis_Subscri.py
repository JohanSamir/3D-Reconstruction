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
    self.pc_pub = message_filters.Publisher('/Dense',PointCloud2,queue_size=2)
    self.pc_sub = message_filters.Subscriber('/kitti/velo/pointcloud', PointCloud2,self.Callback)
    #self.pc_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
    #self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pc_sub], 10, 0.1, allow_headerless=False)
    #self.ts.registerCallback(self.callback)
    
  def callback(self, pointcloud):
    #cont = image.header.seq
    pc_pub.publish(pointcloud)
    
    
def main(args):
  print('Starting...')
  rospy.init_node('sync_node', anonymous=True)
  rospy.loginfo("sync_node on")
  sc = synchronizer()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  
    main(sys.argv)
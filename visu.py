import pandas as pd
import numpy as np
from open3d import *

print("Load a ply point cloud, print it, and render it")
#pcd = read_point_cloud("Dataset_pcd/FinalPoint.pcd")
pcd = read_point_cloud("Dataset_pcd/Hokuyo_6.pcd")
#pcd = read_point_cloud("Dataset_pcd/Hokuyo_2.pcd")

#print(pcd)
#print(np.asarray(pcd.points))
draw_geometries([pcd])


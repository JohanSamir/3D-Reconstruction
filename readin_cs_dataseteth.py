import pandas as pd
import numpy as np
from open3d import *



#:::::::::::::::::::::::::::::::::::
num = 10
# Reading a CSV file
a = pd.read_csv('/home/johan/repos/GitHub/3D-Reconstruction/Dataset_csv/Hokuyo_'+str(num)+'.csv') 
print ('a.shape',a.shape)
b = a.loc[:,"x":"z"]
print (b.shape, type(b))
#Saving a CSV file
#b.to_csv('/home/johan/repos/GitHub/3D-Reconstruction/Hokuyo_0_xyz.csv',index=False)
#:::::::::::::::::::::::::::::::::::
# Converting to Numpy file
c = np.array(b)
print ('c',type(c), c.shape)
#:::::::::::::::::::::::::::::::::::
# Saving the PointCloud in PC
pcd = PointCloud()
pcd.points = Vector3dVector(c)
write_point_cloud('/home/johan/repos/GitHub/3D-Reconstruction/Dataset_pcd/Hokuyo_'+str(num)+'.pcd', pcd)
#:::::::::::::::::::::::::::::::::::
# Load saved point cloud and visualize it
pcd_load = read_point_cloud('/home/johan/repos/GitHub/3D-Reconstruction/Dataset_pcd/Hokuyo_'+str(num)+'.pcd')
draw_geometries([pcd_load])
#:::::::::::::::::::::::::::::::::::


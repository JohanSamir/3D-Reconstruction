import pykitti
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
import os


'''Image and Sparse Laser Fusion'''

basedir = '/home/johan/Desktop/UAO Projects_2/KitiiDatabase'
date = '2011_09_26'
drive = '0005'
#drive = '0001'

dataset = pykitti.raw(basedir, date, drive)

# Find number of PointClouds
point_dir = '/home/johan/Desktop/UAO Projects_2 final/KitiiDatabase/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data'
#se crea un vector vacio para guardar los nombres de las imagenes.
file_list = []
#se obtienen todos los nombre de las imagenes en la carpeta train que tengan extension .csv
file_glob = os.path.join(point_dir, '*.bin')
#se organizan en el vector file_list todos las direcciones de las imagenes 
#encontrados dentro de la carpeta train
file_list.extend(gfile.Glob(file_glob))
file_list = np.sort(file_list)

for cont in range (0,len(file_list)):

	if cont<9:
		numid = "0000" 
		#print ('a')
	elif cont<99:	
		numid = "000"
	elif cont<999:
		numid = "00"
	elif cont<9999:
		num = "0"

	nombre_point = "points"+str(numid)+str(cont)+".csv"
	Velopoints = dataset.get_velo(cont) #dataset.get_velo(15)
	#print(type(Velopoints), next(iter(Velopoints)))
	#print(Velopoints[0],Velopoints[1])
	Velopoints = np.asarray(Velopoints, np.float32)
	#Subsamling, Sparse
	Velopoints = Velopoints[::2]
	#Pointins junt in FRONT of the CAMERA
	#idx = Velopoints[:,0]<5
	#Velopoints = np.delete(Velopoints, np.where(idx),0)
	Velopoints = Velopoints[:,0:3]
	print(Velopoints.shape, Velopoints)
	np.savetxt("/home/johan/Desktop/UAO Projects_2 final/KitiiDatabase/PointCloud_CSV_kitti_1/"+nombre_point, Velopoints,  header='x,y,z', delimiter=',',comments='')
	#print(type(Velopoints3P),Velopoints3P.shape)
	#Velopoints3P_3D_Plot = Velopoints3P
	print(cont)
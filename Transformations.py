import pandas as pd
import numpy as np
from open3d import *

#:::::::::::::::::::::::::::::::::::
#0-1
T0_1 = np.array([[0.997681,	-0.0680117,	-0.00256759,	0.0867364],
				[0.0680267,	0.997663,	0.00630216	,-0.00310835],
				[0.00213301,	-0.00646222	,0.999977,	-0.0174408]])
#:::::::::::::::::::::::::::::::::::
T1_2 = np.array([[0.999984,	0.0035861,	0.00434511,	0.0154002],
				[-0.00358643,	0.999994,	7.03E-05,	0.05292],
				[-0.00434484,	-8.59E-05,	0.999991,	-0.00493765]])
#:::::::::::::::::::::::::::::::::::
T2_3  = np.array([[0.999984,	0.0035861,	0.00434511,	0.0154002],
				 [-0.00358643,	0.999994,	7.03E-05,	0.05292],
				 [-0.00434484,	-8.59E-05,	0.999991,	-0.00493765]])

#:::::::::::::::::::::::::::::::::::
T3_4= np.array([[0.999984,	0.0035861,	0.00434511	,0.0154002],
				[-0.00358643,	0.999994,	7.03E-05,	0.05292],
				[-0.00434484,	-8.59E-05,	0.999991,	-0.00493765]])
#:::::::::::::::::::::::::::::::::::
T4_5 = np.array([[0.982251	,0.187496	,0.0053248	,0.564834],
				[-0.187518	,0.982253,	0.0041074	,-0.173654],
				[-0.00446019	,-0.00503299,	0.999977,	-0.00322318]])
#:::::::::::::::::::::::::::::::::::
T5_6 = np.array([[0.982251,	0.187496	,0.0053248	,0.564834],
				[-0.187518,	0.982253,	0.0041074,	-0.173654],
				[-0.00446019,	-0.00503299,	0.999977,	-0.00322318]])
#:::::::::::::::::::::::::::::::::::

T1_2F = T0_1 * T1_2
T2_3F = T0_1 * T1_2 * T2_3
T3_4F = T0_1 * T1_2 * T2_3 * T3_4
T4_5F = T0_1 * T1_2 * T2_3 * T3_4 * T4_5
T5_6F = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6


PointCloudFinal = np.array([])

for i in range (0,7):
	
	# command line arguments are stored in the form 
	# of list in sys.argv 
	num = raw_input(" Enter PointCloud ")

	#:::::::::::::::::::::::::::::::::::
	# Reading a CSV file
	a = pd.read_csv('/home/johan/repos/GitHub/3D-Reconstruction/Dataset_csv/Hokuyo_'+str(num)+'.csv') 
	#print ('a.shape',a.shape)
	b = a.loc[:,"x":"z"]
	print ('b',b.shape, type(b))

	# Converting to Numpy file
	g = np.array(b)
	#print ('shape:',c.shape[1])
	f = np.ones((g.shape[0],1))
	#print ('f:',f.shape)

	c = np.hstack((g,f))
	#print ('shape:',c.shape[1])

	c = np.transpose(c)
	print ('c',type(c), c.shape)
	#:::::::::::::::::::::::::::::::::::
	
	if i == 0:
		PointCloudFinal = np.transpose(g)
		print ('PointCloudFinal ',PointCloudFinal.shape)

	elif i == 1:
		c = np.dot(T0_1,c)
		#print ('PointCloudFinal--C ',c.shape)
		#PointCloudFinal = np.append(PointCloudFinal, c)
		PointCloudFinal = np.concatenate((PointCloudFinal, c), axis=1)
		print ('PointCloudFinal ',PointCloudFinal.shape)

	elif i == 2:
		c = np.dot(T1_2F,c)
		PointCloudFinal = np.concatenate((PointCloudFinal, c), axis=1)
		print ('PointCloudFinal ',PointCloudFinal.shape)

	elif i == 3:
		c = np.dot(T2_3F,c)
		PointCloudFinal = np.concatenate((PointCloudFinal, c), axis=1)
		print ('PointCloudFinal ',PointCloudFinal.shape)

	elif i == 4:
		c = np.dot(T3_4F,c)
		PointCloudFinal = np.concatenate((PointCloudFinal, c), axis=1)
		print ('PointCloudFinal ',PointCloudFinal.shape)

	elif i == 5:
		c = np.dot(T4_5F,c)
		PointCloudFinal = np.concatenate((PointCloudFinal, c), axis=1)
		print ('PointCloudFinal ',PointCloudFinal.shape)

	elif i == 6:
		c = np.dot(T5_6F,c)
		PointCloudFinal = np.concatenate((PointCloudFinal, c), axis=1)
		print ('PointCloudFinal ',PointCloudFinal.shape)

	# Saving the PointCloud in PC

c = np.transpose(c)
print ('c',type(c), c.shape)

pcd = PointCloud()
pcd.points = Vector3dVector(c)
write_point_cloud('/home/johan/repos/GitHub/3D-Reconstruction/Dataset_pcd/FinalPoint.pcd', pcd)
np.savetxt('home/johan/repos/GitHub/3D-Reconstruction/Dataset_pcd/FinalPoint.csv', c)

#:::::::::::::::::::::::::::::::::::
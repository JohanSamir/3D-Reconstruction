import numpy as np
#from open3d import *
import pypcd
import os
import subprocess


if __name__ == '__main__':

	#------- rutas hacia las nubes de puntos y el ejecutable----------
	# .cpp file: addres /home/johan/Libraries/libpointmatcher/examples
	icp_path="/home/johan/Libraries/libpointmatcher/build/examples"
	pointcloud1_path="/home/johan/repos/GitHub/3D-Reconstruction/Dataset_csv/pointscsv1.csv"
	pointcloud2_path="/home/johan/repos/GitHub/3D-Reconstruction/Dataset_csv/pointscsv137.csv"
	
	#--------------- correr el ejecutable de c++ ------------------
	#runicp = subprocess.Popen([os.path.join(icp_path,"icp_simple"),pointcloud1_path,pointcloud2_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	runicp = subprocess.Popen([os.path.join(icp_path,"icp_simple"),pointcloud1_path,pointcloud2_path,'HI'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	runicp.wait()

	#-------------leer los datos de devuelve en el terminal------
	stdout, stderr = runicp.communicate()
	print('stdout:',stdout)
	#print('stderr:',stderr)

	#----------------Guadar los datos en txt---------------------
	'''
	file = open("logs.txt","a")
	file.write("-------------------------------------\n")
	#simplejson.dump(maxiooas, file)
	file.write(str(stdout))
	file.write("-------------------------------------\n")
	file.close()
	#print(np.char.split(stdout, sep = ' '))
	'''

    	


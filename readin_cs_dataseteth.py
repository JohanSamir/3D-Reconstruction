import pandas as pd

# Reading a CSV file
a = pd.read_csv('/home/johan/repos/GitHub/3D-Reconstruction/Hokuyo_0.csv') 
print ('a.shape',a.shape)
b = a.loc[:,"x":"z"]
print (b.shape)
#Saving a CSV file
b.to_csv('/home/johan/repos/GitHub/3D-Reconstruction/Hokuyo_0_xyz.csv',index=False)



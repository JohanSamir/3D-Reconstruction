# coding: utf-8

# # Velodyne Points Projected into Image
# In[3]:
import pykitti
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.segmentation import slic,mark_boundaries
from skimage import exposure

from segraph import create_graph
from random import randint

import scipy.sparse.linalg
from scipy.sparse.linalg import cgs


def depth_laser_camer(file_list,idpoint,ns):
    '''Image and Sparse Laser Fusion'''

    basedir = '/home/johan/Desktop/UAO Projects_2/KitiiDatabase'
    date = '2011_09_26'
    drive = '0005'
    #drive = '0001'
    tr = file_list
    print(tr)
    dataset = pykitti.raw(basedir, date, drive)
    img = plt.imread(tr)
    Velopoints = dataset.get_velo(idpoint) #67, image 67

    #Loading the calibration parameters
    P_rect_20 = dataset.calib.P_rect_20
    R_rect_20 = dataset.calib.R_rect_20
    T_rect_20 = dataset.calib.T_cam0_velo_unrect

    P_rect_20 = np.matrix(P_rect_20)
    R_rect_20 = np.matrix(R_rect_20)
    T_rect_20 = np.matrix(T_rect_20)

    Velopoints = np.asarray(Velopoints, np.float32)
    T1 = (P_rect_20 * R_rect_20)* T_rect_20

    # In[5]:

    Velopoints = Velopoints[::2]
    Velopoints = np.delete(Velopoints, np.where(Velopoints[:,0]<5),0)
    Velopoints3P = Velopoints[:,0:3]

    Velopoints_getvalues = Velopoints3P
    Velopoints3P_3D_Plot = Velopoints3P

    # In[6]:


    dim_M = T1.shape[0]
    if Velopoints3P.shape[1] < T1.shape[1]:
        ones_vect_Velo3p = np.ones ((Velopoints3P.shape[0], 1),int)
        Velopoints3P = np.concatenate ((Velopoints3P, ones_vect_Velo3p), axis = 1)

    Velopoints3P = np.matrix(Velopoints3P)
    Velopoints3PP = Velopoints3P

    # In[7]:

    y = np.transpose((T1 * np.transpose(Velopoints3P)))
    x_y = y[:,0:dim_M-1]
    b_ones = np.ones((1, dim_M-1),int)
    z = y[:,dim_M-1]
    p_out = np.divide(x_y, np.multiply(z , b_ones))
    idx_p_out = np.arange(0,p_out.shape[0], 1)

    x = p_out[:,0]
    y = p_out[:,1]

    # In[8]:

    img_x = img.shape[1]
    img_y = img.shape[0]

    pointxt = np.logical_or(x < 0, x > img_x)
    idex = np.where(np.logical_not(pointxt))[0]
    pout_ft = p_out[idex,:]

    #Index of point clouds
    idx_p_outt = idx_p_out[idex]
    xa = pout_ft[:,0]
    yb = pout_ft[:,1]
    pointyt = np.logical_or(yb < 0,yb > img_y)

    idexy = np.where(np.logical_not(pointyt))[0]
    pout_fty =pout_ft[idexy,:]
    idx_p_outtt = idx_p_outt[idexy]

    xaa = pout_fty[:,0]
    ybb = pout_fty[:,1]

    pout_fty = np.matrix.round(pout_fty)
    pout_fty = pout_fty.astype(int)

    # In[9]:

    matrix_points = np.zeros((img.shape[0], img.shape[1]))

    for i in range(0,pout_fty.shape[0]):
        x = pout_fty[i,0] 
        y = pout_fty[i,1]
        ppoints = Velopoints3PP[idx_p_outtt[i],0:3]
           
        matrix_points[y-1,x-1] = np.sqrt(np.power( ppoints[:,0], 2)+ np.power( ppoints[:,1], 2) + np.power(ppoints[:,2], 2))

    # In[10]:


    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    # # Image and Sparse Laser Fusion

    # In[11]:

    image = img
    segments = slic(image, n_segments = ns, enforce_connectivity = True)
    vertices, edges = create_graph(segments)

    # Compute centers:
    gridx, gridy = np.mgrid[:segments.shape[0], :segments.shape[1]]
    centers = dict()
    for v in vertices:
        centers[v] = [round(gridy[segments == v].mean()), round(gridx[segments == v].mean())]

    # In[12]:

    '''
    Getting the branches' values according to the pixel postision (close to the edges, corners, 
    or inside the image).
    Here the value of 0° is respresentes as 360°
    '''

    com =  np.array(list(range(len(centers)))).reshape(1,len(centers))
    b_grafo = np.zeros(len(centers))
    orientacion = dict()
    q = {}
    q_blue = {}
    q_red= {}
    q_gren= {}
    pe = {}
    vref =  np.array([])
    average_pixel = dict()
    average_pix_blue = dict()
    average_pix_red = dict()
    average_pix_green = dict()
    average_points = dict()

    #-----------------------
    #4 corners
    #vref: it is to compare what segments were assigment to some coordinates
    a_rup = segments[0,gray_image.shape[1]-1]
    b_grafo[a_rup] = 2
    vref = np.append(vref,a_rup)
    orientacion[a_rup] = [180, -90]

    a_lup = segments[0,0]
    vref = np.append(vref,a_lup)
    b_grafo[a_lup] = 2
    orientacion[a_lup] = [-90,0]

    a_ldown = segments[gray_image.shape[0]-1,0]
    vref = np.append(vref,a_ldown )
    b_grafo[a_ldown] = 2
    orientacion[a_ldown] = [90,0]

    a_rdown =  segments[gray_image.shape[0]-1, gray_image.shape[1]-1]
    vref = np.append(vref,a_rdown)
    b_grafo[a_rdown] = 2
    orientacion[a_rdown] = [90,180]

    #print(len(centers))
    #print(vref)

    # In[13]:


    #Continuation of the cell above
    #------------------------
    for j in range(0,gray_image.shape[0]):
        for i in range(0,gray_image.shape[1]):
            #segments[y][x]
            #gray_image.shape[0] ---> y (j)
            #gray_image.shape[1] ---> x (i)
            
            s = segments[j,i]
            ta = s == vref
            #print ta
            #print a,type(a)
            if ta.any() == False:
               
                if j == 0 and i > 0 and i < gray_image.shape[1]-1:
                    vref = np.append(vref,s)
                    #Vertical Line_1
                    b_grafo[s] = 3
                    orientacion[s] = [180,-90,0] 
                elif j == gray_image.shape[0]-1 and i > 0 and i < gray_image.shape[1]-1:
                    #Vertical Line_2
                    vref = np.append(vref,s)
                    b_grafo[s] = 3
                    orientacion[s] = [90,180,0]
                elif i == 0 and j > 0 and j < gray_image.shape[0]-1:
                    #Vertical Line_1
                    vref = np.append(vref,s)
                    b_grafo[s] = 3
                    orientacion[s] = [90,-90,0] 
                elif i == gray_image.shape[1]-1 and j > 0 and j < gray_image.shape[0]-1:
                    #Vertical Line_1
                    vref = np.append(vref,s)
                    b_grafo[s] = 3
                    orientacion[s] = [90,180,-90]
                else:
                    vref = np.append(vref,s)
                    b_grafo[s] = 4
                    orientacion[s] = [90,180, -90, 0]
                 
            a = segments[j,i] == com
            #print a,type(a)
            af = np.where(a)[True]
            #example.setdefault(af[0], []).append(cc[i,j])
            q.setdefault(af[0], []).append(gray_image[j][i])    
            q_blue.setdefault(af[0], []).append(blue[j][i])
            q_red.setdefault(af[0], []).append(green[j][i])
            q_gren.setdefault(af[0], []).append(red[j][i])
            
            qe = matrix_points[j][i]
            if qe != 0:
                pe.setdefault(af[0], []).append(qe)

    print('Ready!')
    #Getting the average values of color and point cloud.


    # In[14]:


    ar = np.array(pe.keys())
    for rt in range (0, len(centers)):
        average_pixel[rt] = sum(q[rt])/len(q[rt])
        average_pix_blue[rt] = sum(q_blue[rt])/len(q_blue[rt])
        average_pix_red[rt] = sum(q_red[rt])/len(q_red[rt])
        average_pix_green[rt] = sum(q_gren[rt])/len(q_gren[rt])

        ta = rt == ar
        #print(ta)
        if ta.any() == False:
            average_points[rt] = 0
        else:
            average_points[rt] = sum(pe[rt])/len(pe[rt])

    # In[15]:

    #Average of the chanels RGB for each segment
    aaverage_pix_blue = np.asarray(average_pix_blue.values())
    aaverage_pix_red = np.asarray(average_pix_red.values())
    aaverage_pix_green = np.asarray(average_pix_green.values())

    RG = np.column_stack((aaverage_pix_red,aaverage_pix_green))
    RGB = np.column_stack((RG,aaverage_pix_blue ))

    # In[16]:

    #print 'b_grafo',b_grafo.shape,'\n',b_grafo
    for i in range (0,len(b_grafo)):
        if b_grafo[i] == 0:
            b_grafo[i] = 4
            orientacion[i] = [90,180, -90, 0]

    # In[17]:

    ''' Image Graph '''
    tunning_pa = 1
    l=0
    edges_pairs = {}
    many_edges_pairs = {}

    for i in range(0,len(centers)):
        t=0
        for edge in edges:
           
            if edge[0]== i or edge[1]== i:
                edges_pairs[l] = edge
                l = l+1
                t = t+1
        many_edges_pairs[i]= t     

    # In[18]:

    '''Calculating the atan2 for every branch in the main graph. Moreover, a data conversion
    was made for getting the values between 0° and 360°'''

    edges_pair_wise = {}
    a = edges_pairs.values()
    f_g = many_edges_pairs[0]
    t = 0
    f = 0
    s = 0
    r = 0
    for edge in a:
        f_g = many_edges_pairs[f]
        #print 'f',f_g
        s = s + 1
        r_s = f_g - s
        #print 'r_s',r_s
        if r_s >= 0:
            if edge[0] == r:
                y = centers[edge[1]][1] - centers[edge[0]][1] 
                x = centers[edge[1]][0] - centers[edge[0]][0] 
                e = (np.arctan2(y,x) * 180 / np.pi)
                #To convert the angle according to righ coordenates.
                #-1 represents the inverse of the Y axle
                if e == 0:
                    e = e 
                else:
                    e = e*-1
                edges_pair_wise[t] = e
                
            else:
                y = centers[edge[0]][1] - centers[edge[1]][1]
                x = centers[edge[0]][0] - centers[edge[1]][0]           
                e = (np.arctan2(y, x) * 180 / np.pi)
                #To convert the angle according to righ coordenates.
                #-1 represents the inverse of the Y axle
                if e == 0:
                    e = e 
                else:
                    e = e*-1        
                edges_pair_wise[t] = e
                #print 'b'
            t = t + 1 
            if r_s == 0:
                f = f + 1
                s = 0
                r = r + 1

    # In[19]:

    '''
    Getting the four neighboors for each pixel in the image.
    '''
    u = -1
    a = 0
    f = many_edges_pairs[0]
    indx_angulos =[]
    angu_test = dict()
    angu_test_idx = dict()

    for i in range (0,len(centers)):
        angulos = []
        angulos_idx = []
        orientacio_nu = orientacion[i]
        b_grafo_nu = int(b_grafo[i])
        many_edges_pairs_num = many_edges_pairs[i]

        for j in range(a,f):
            u = u + 1
            angulos = np.append(angulos,edges_pair_wise[j])
            angulos_idx = np.append(angulos_idx,u)
        l = np.array(angulos)
        
        angu_test[i] = l
        angu_test_idx[i] = angulos_idx
        a = many_edges_pairs_num + a
        if i < len(centers)-1:
            f = a + many_edges_pairs[i+1]

    # In[20]:

    graph = []
    graph_dict_many = dict()
    graph_dict = dict()
    for i in range (0,len(centers)):
        many_edges_pairs_num = many_edges_pairs[i]
        orientacio_nu = orientacion[i]
        b_grafo_nu = int(b_grafo[i])
        u = []
        
        for z in range(0,b_grafo_nu):
            t = orientacio_nu[z]
            a = []
            for s in range (0,many_edges_pairs_num):
                #print 's',s
                if t == -90: 
                    p = abs(t-angu_test[i][s])
                    a = np.append(a,p)
                elif t == 0:
                    p = abs(t-angu_test[i][s])
                    a = np.append(a,p)
                elif t == 90:
                    p = abs(t-angu_test[i][s])
                    a = np.append(a,p)
                elif t == 180:
                    p = abs(t-abs(angu_test[i][s]))
                    a = np.append(a,p)
            min_value = np.argmin(a)
            min_value = angu_test_idx[i][min_value]
            coorde_min_value = edges_pairs[min_value]
            u = np.append(u, coorde_min_value)
            graph = np.append(graph,coorde_min_value)
        
        graph_dict[i] = u 
        u = len(u)
        graph_dict_many[i] = u
                   
    euler = np.zeros((len(graph),1))/2

    # In[21]:

    #Test
    for i in range (0, len(graph_dict)):
         if graph_dict_many[i]>8:
            pass

    # # Cost function with Data Cost and Discontinuity Cost

    # In[23]:

    p = 0
    r = 0
    measure_confi = 1
    diagonal = np.zeros((len(centers),1))
    V_points = np.zeros((len(centers),1))

    for i in range(0,len(centers)):
                    
            V_points[i] = average_points[i]
            
            if average_points[i] > 0:
                diagonal[i] = measure_confi
            else:
                diagonal[r] = 0
                       
    x = np.eye(diagonal.shape[0])
    diagonal = x * diagonal
    #print 'Vector Points cloud: ',V_points.shape
    #print 'W Matrix: (Diagonal)',diagonal.shape
    #np.savetxt('V_points'+'.csv', V_points, delimiter=',') 

    # In[24]:

    '''Adjencia Matrix'''
    Ad_m = np.zeros((sum(graph_dict_many.values())/2,len(centers)), dtype = float)
    d = 0
    k = 0
    euler = np.zeros((len(edges_pairs),1))
    f=0

    for i in range (0,len(centers)):
        for j in range (0,len(graph_dict[i]),2):
            a = j+1
            if graph_dict[i][j]== i:
                Ad_m[f][int(graph_dict[i][j])] = 1
                Ad_m[f][int(graph_dict[i][a])] = -1
                
                                    
            elif graph_dict[i][a] == i:
                Ad_m[f][int(graph_dict[i][a])] = 1
                Ad_m[f][int(graph_dict[i][j])] = -1
            
            f=f+1 
            
    # In[25]:

    '''In this part of the code, we calculated S.
    S = E * Ad_m where E = [It is a vector of differece in pixel appereance] and Ad_m = [it is a Adjacency matrix]
    '''
    e = []
    tunning_pa = 0.02
    for i in range (0,len(centers)):
        for j in range (0,len(graph_dict[i]),2):
            a = j+1
            if graph_dict[i][j]== i:
                m1 = np.power((average_pixel[graph_dict[i][j]]-average_pixel[graph_dict[i][a]]),2)
                m1 = (m1 / tunning_pa) * (-1)
                m1 = np.exp(m1)  
                e = np.append(e, m1)
                              
            elif graph_dict[i][a] == i:
                m1 = np.power((average_pixel[graph_dict[i][a]]-average_pixel[graph_dict[i][j]]),2)
                m1 = (m1 / tunning_pa) * (-1)
                m1 = np.exp(m1)                
                e = np.append(e, m1)      

    # In[26]:

    '''In this part of the code, we calculated S.
    S = E * Ad_m where E = [It is a vector of differece in pixel appereance] and Ad_m = [it is a Adjacency matrix]
    '''
    s = []
    p = 0

    for i in range (0,Ad_m.shape[0]):
        for j in range (0,Ad_m.shape[1]):
            
            if Ad_m[i,j] == 1:
                Ad_m[i,j] = Ad_m[i][j]*e[p]    
            
            elif Ad_m [i][j] == -1:
                Ad_m [i][j] = Ad_m [i][j]*e[p]        

    # In[27]:

    lamda_1 = 0.4
    lamda_2 = 1

    SS = np.transpose(Ad_m)
    SS = np.matmul(SS,Ad_m)
    #print('Matrix S:\n',SS.shape)

    WW = np.transpose(diagonal)
    WW = np.matmul(WW,diagonal)
    #print('Matrix W:\n', WW.shape)

    A = (((lamda_1*lamda_2)*SS)+((1-lamda_1)*WW))/(1-lamda_1)
    #print('Matrix A:\n',A.shape)

    B = np.matmul(WW,V_points)
    #print('Matrix B:\n',B.shape)

    x_values_inf = cgs(A, B)
    #print('Matrix x (Inference):\n',x_values_inf ,len(x_values_inf [0]),type(x_values_inf ))
    #np.savetxt('x inference'+'.csv', x_values_inf [0], delimiter=',')   


    # In[31]:

    r = []
    t = []
    k = []
    r1 = []
    t1 = []
    k1 = []

    for i in range (0, len(centers)):
        
        if V_points[i] != 0:
            c = int(centers[i][0])
            r = np.append(r, c)
            r = r.tolist()
            d = int(centers[i][1])
            t = np.append(t, d)
            t = t.tolist()
            f = V_points[i]
            k = np.append(k,f)
            k = k.tolist()
                    
        c1 = int(centers[i][0])
        r1 = np.append(r1, c1)
        r1 = r1.tolist()
        d1 = int(centers[i][1])
        t1 = np.append(t1, d1)
        t1 = t1.tolist()
        f1 = x_values_inf[0][i]
        k1 = np.append(k1,f1)
        k1 = k1.tolist()   

    # In[31]:

    #------------------------
    uu_o = np.zeros((image.shape[0]*image.shape[1]))
    xx = np.zeros((image.shape[0]*image.shape[1]))
    yy = np.zeros((image.shape[0]*image.shape[1]))

    uu_mat = np.zeros((image.shape[0],image.shape[1]))
    print uu_mat.shape
    t = 0
    for j in range(0,gray_image.shape[0]):
        for i in range(0,gray_image.shape[1]):
                   
            s = segments[j,i]
            uu_mat[j,i] = k1[s]
            xx[t] = i
            yy[t] = j
            uu_o[t] = k1[s]
            t = t+1   
    print uu_o.shape

    #print (uu_mat.shape, type(uu_mat))
    return(uu_mat)

import cv2
import numpy as np

#filepath
Points3d_A_path = 'Source/points3d_A.txt'
Points3d_B_path = 'Source/points3d_B.txt'
Points2d_A_path = 'Source/points2d_A.txt'
Points2d_B_path = 'Source/points2d_B.txt'

#Interior parameters
K = np.array([(746.07,0,493.94),(0,743.92,488.76),(0,0,1)],dtype=float)
fx = K[0][0]
fy = K[1][1]
cx = K[0][2]
cy = K[1][2]

#Transform to numpy array
Points3d_A = np.loadtxt(Points3d_A_path,dtype=float)
Points3d_B = np.loadtxt(Points3d_B_path,dtype=float)
Points2d_A = np.loadtxt(Points2d_A_path,dtype=float)
Points2d_B = np.loadtxt(Points2d_B_path,dtype=float)

#Transform to Homogeneous
Homogeneous_addarray = np.ones(Points3d_A.shape[0],dtype=float)
Points3d_A_Homo = np.c_[Points3d_A,Homogeneous_addarray]
Points3d_B_Homo = np.c_[Points3d_B,Homogeneous_addarray]
Points2d_A_Homo = np.c_[Points2d_A,Homogeneous_addarray]
Points2d_B_Homo = np.c_[Points2d_B,Homogeneous_addarray]


#Declaration:   GotA (Ax = 0)
#input:         Single Points3d Points2d
#output:        A
def GotA(Points3d,Points2d):
    xFx = Points3d[0]*fx
    yFx = Points3d[1]*fx
    zFx = Points3d[2]*fx
    Fx  = fx
    A11 = np.array([xFx,yFx,zFx,Fx],dtype=float)
    A12 = np.zeros(4,dtype=float)
    xCx_ux = Points3d[0]*cx - Points2d[0]*Points3d[0]
    yCx_uy = Points3d[1]*cx - Points2d[0]*Points3d[1]
    zCx_uz = Points3d[2]*cx - Points2d[0]*Points3d[2]
    Cx_u   = cx - Points2d[0]
    A13 = np.array([xCx_ux,yCx_uy,zCx_uz,Cx_u],dtype=float)
    A21 = np.zeros(4,dtype=float)
    xFy = Points3d[0]*fy
    yFy = Points3d[1]*fy
    zFy = Points3d[2]*fy
    Fy  = fy
    A22 = np.array([xFy,yFy,zFy,Fy],dtype=float)
    xCy_vx = Points3d[0]*cy - Points2d[1]*Points3d[0]
    yCy_vy = Points3d[1]*cy - Points2d[1]*Points3d[1]
    zCy_vz = Points3d[2]*cy - Points2d[1]*Points3d[2]
    Cy_v   = cy - Points2d[1]
    A23 = np.array([xCy_vx,yCy_vy,zCy_vz,Cy_v],dtype=float)
    A1 = np.r_[A11,A12,A13].reshape(1,12)
    A2 = np.r_[A21,A22,A23].reshape(1,12)
    A  = np.c_[A1,A2].reshape(2,12)
    return A

#Declaration:   GotRT by SVD (3*4) [3*3 + 3*1]
#input:         Single Points3d Points2d
#output:        RT
def GotRT(Points3d_Homo,Points2d_Homo):
    length = Points3d_Homo.shape[0]
    for i in range(length):
        if (i == 0):
            A = GotA(Points3d_Homo[i], Points2d_Homo[i])
        else:
            A = np.concatenate((A, GotA(Points3d_Homo[i], Points2d_Homo[i])), axis=0)
    U, Sigma, Vt = np.linalg.svd(A)
    x = Vt[-1]
    RT = Vt[-1].reshape(3, 4)
    return RT



if __name__ =='__main__':
    RT_A = GotRT(Points3d_A_Homo,Points2d_A_Homo)
    RT_B = GotRT(Points3d_B_Homo,Points2d_B_Homo)
    #print RT martrix
    #print(RT_A)
    #print(RT_B)

    #calculate the 3d to 2d predict
    PredictA_2Dlist=[]
    PredictB_2Dlist=[]
    for i in range(len(Points3d_A_Homo)):
        PredictA_2D = np.dot(K,np.dot(RT_A,Points3d_A_Homo[i])).reshape(1,3)
        PredictA_2D = PredictA_2D / PredictA_2D[0][2]
        PredictA_2Dlist.append(PredictA_2D)
        PredictB_2D = np.dot(K,np.dot(RT_B,Points3d_B_Homo[i])).reshape(1,3)
        PredictB_2D = PredictB_2D / PredictB_2D[0][2]
        PredictB_2Dlist.append(PredictB_2D)
    #save results if necessary[default fold: Source]
    PredictA_2Darray = np.array(PredictA_2Dlist)
    PredictB_2Darray = np.array(PredictB_2Dlist)
    fpA = open('Source/pointspredictA.txt','w')
    fpB = open('Source/pointspredictB.txt','w')
    for i in range(len(Points3d_A_Homo)):
        fpA.write(str(int(PredictA_2Darray[i][0][0]))+" "+str(int(PredictA_2Darray[i][0][1]))+"\n")
        fpB.write(str(int(PredictB_2Darray[i][0][0]))+" "+str(int(PredictB_2Darray[i][0][1]))+"\n")
    fpA.close()
    fpB.close()

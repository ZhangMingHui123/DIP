import cv2
import numpy as np

#File and Data path
FilepathA = 'Source/A.png'
FilepathB = 'Source/B.png'
txtpathA = 'Source/points2d_A.txt'
txtpathB = 'Source/points2d_B.txt'

#Transform to numpy array
Points2d_A = np.loadtxt(txtpathA, dtype=int)
Points2d_B = np.loadtxt(txtpathB, dtype=int)

#Declaration:   Show2Dnodes
#input:         Points2d_A Points2d_B FilepathA FilepathB
#output:        None
def Show2Dnodes(Points2d_A,Points2d_B,FilepathA,FilepathB):
    Points2d_A_list = Points2d_A.tolist()
    Points2d_B_list = Points2d_B.tolist()
    Image_A = cv2.imread(FilepathA)
    Image_B = cv2.imread(FilepathB)
    #line argument
    point_size = 2
    point_color = (0, 0, 255)
    thickness = 4
    for point in Points2d_A_list:
        point_tuple = tuple(point)
        cv2.circle(Image_A, point_tuple, point_size, point_color, thickness)
    for point in Points2d_B_list:
        point_tuple = tuple(point)
        cv2.circle(Image_B, point_tuple, point_size, point_color, thickness)
    Image_A_resize = cv2.resize(src=Image_A, dsize=(0,0),fx=0.5,fy=0.5)
    Image_B_resize = cv2.resize(src=Image_B, dsize=(0,0),fx=0.5,fy=0.5)
    cv2.imshow("Image_A_resize", Image_A_resize)
    cv2.imshow("Image_B_resize", Image_B_resize)
    cv2.waitKey(0)

#Declaration:   Slove affine M by opencv's function
#input:         Points2d_A Points2d_B
#output:        affine M
def SolveAffineM_byopencv(Points2d_A,Points2d_B):
    M = cv2.estimateAffine2D(Points2d_A, Points2d_B, ransacReprojThreshold=67)
    #M = cv2.estimateAffine2D(Points2d_A, Points2d_B)
    return M[0]

#Declaration:   GetA_B A:2*6 B:2*1 for each point
#input:         Single Points2d_A Points2d_B
#output:        A B
def GetA_B(Points2d_A,Points2d_B):
    A = np.array([[Points2d_A[0],Points2d_A[1],1,0,0,0],[0,0,0,Points2d_A[0],Points2d_A[1],1]]).reshape(2,6)
    B = np.array([[Points2d_B[0]],[Points2d_B[1]]])
    return A,B

#Declaration:   Slove affine M by lstsq method
#input:         Points2d_A Points2d_B
#output:        affine M
def SolveAffineM_bylstsq(Points2d_A,Points2d_B):
    for i in range(len(Points2d_A)):
        if (i == 0):
            A, B = GetA_B(Points2d_A[i], Points2d_B[i])
        else:
            tempA, tempB = GetA_B(Points2d_A[i], Points2d_B[i])
            A = np.concatenate((A, tempA), axis=0)
            B = np.concatenate((B, tempB), axis=0)
    M = (np.linalg.lstsq(a=A, b=B, rcond=None)[0]).reshape(2,3)
    return M

#Declaration:   Affine one Image
#input:         srcFilepath M
#output:        dstImage(after affine)
def AffineImage(srcFilepath,M):
    srcImage = cv2.imread(srcFilepath)
    dsize = (srcImage.shape[0],srcImage.shape[1])
    dstImage = cv2.warpAffine(src=srcImage,M=M,dsize=dsize,borderValue=0)
    return dstImage


if __name__ == '__main__':
    Show2Dnodes(Points2d_A=Points2d_A,Points2d_B=Points2d_B,FilepathA=FilepathA,FilepathB=FilepathB)
    M1 = SolveAffineM_byopencv(Points2d_A=Points2d_A,Points2d_B=Points2d_B)
    print("使用opencv自带的函数求到的M:\n"+str(M1))
    M2 = SolveAffineM_bylstsq(Points2d_A=Points2d_A,Points2d_B=Points2d_B)
    print("使用最小二乘法求到的M:\n"+str(M2))
    dstImage = AffineImage(srcFilepath=FilepathA,M=M1)
    dstImage_resize = cv2.resize(src=dstImage,dsize=(0,0),fx=0.5,fy=0.5)
    #imshow or save if necessary
    cv2.imshow("affine", dstImage_resize)
    cv2.imwrite('Source/A_affine.png',dstImage)
    cv2.waitKey(0)
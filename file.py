import cv2
import numpy as np

src=cv2.imread("1.png",0)
row,col=src.shape
ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#Thresholding
p1=np.zeros((src.shape[0],src.shape[1])) #Declaration
output = cv2.connectedComponentsWithStats(thresh)

p2=np.array([255 if output[1][i,j]==0 else 0 for i in range(row) for j in range(col)],np.uint8).reshape(row,col)
cv2.imshow("1",p2)
cv2.waitKey(0)
##practica 10
import numpy as np
from matplotlib import pyplot as plt
import math as ma
import cv2 #opencv

ima = cv2.imread("gato.png")
cv2.imshow("Original",ima)



# Select ROI
r = cv2.selectROI("select the area", ima)

imCrop = ima[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imwrite("imageROI.jpg", imCrop)

mask = np.zeros(ima.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#height, width, channels = img.shape

rect =(int(r[0]),int(r[1]), int(r[2]),int(r[3]))

cv2.grabCut(ima,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = ima*mask2[:,:,np.newaxis]

nuev=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(nuev)
plt.axis('off')
plt.savefig('imageOFF.jpg')
plt.axis('on')
plt.colorbar()
plt.show()

# Corners
img = cv2.imread('imageOFF.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imwrite("imageCORNERS.jpg", img)
cv2.imshow('Corner',img)

cv2.waitKey()
cv2.destroyAllWindows()


from PIL import Image
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

'''
img = cv2.imread("hyper_denoised_binary_image.png", 0)


# convert image to grayscale image
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# convert the grayscale image to binary image
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh,1,2)

cnt = contours[0]
'''

gwash = cv2.imread("hyper_denoised_binary_image.png") #import image
gwashBW = cv2.cvtColor(gwash, cv2.COLOR_BGR2GRAY) #change to grayscale


# plt.imshow(gwashBW, 'gray') #this is matplotlib solution (Figure 1)
# plt.xticks([]), plt.yticks([])
# plt.show()

cv2.imshow('gwash', gwashBW) #this is for native openCV display
cv2.waitKey(0)

ret,thresh1 = cv2.threshold(gwashBW, 1, 255, cv2.THRESH_BINARY) #the value of 15 is chosen by trial-and-error to produce the best outline of the skull
kernel = np.ones((3, 3),np.uint8) #square image kernel used for erosion
erosion = cv2.erode(thresh1, kernel,iterations = 1) #refines all edges in the binary image

opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image

# plt.imshow(closing, 'gray') #Figure 2
# plt.xticks([]), plt.yticks([])
# plt.show()

contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #find contours with simple approximation

cv2.imshow('cleaner', closing) #Figure 3
cv2.drawContours(closing, contours, -1, (100, 100, 100), 4)
cv2.waitKey(0)

areas = [] #list to hold all areas

for contour in contours:
  ar = cv2.contourArea(contour)
  areas.append(ar)

max_area = max(areas)
max_area_index = areas.index(max_area) #index of the list element with largest area

cnt = contours[max_area_index] #largest area contour
 
# calculate moments of binary image
M = cv2.moments(cnt)
print M
 
# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
 

# put text and highlight the center
cv2.circle(gwash, (cX, cY), 5, (120, 20, 60), -1)
cv2.putText(gwash, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 20, 60), 2)
 
# display the image
cv2.imshow("Image", gwash  )
cv2.waitKey(0)


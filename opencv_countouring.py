import numpy as np
import cv2 #this is the main openCV class, the python binding file should be in /pythonXX/Lib/site-packages
from matplotlib import pyplot as plt

def outline(image, threshold, iteration, kernel_size, maxlevel):
	gwash = cv2.imread(image) #import image
	gwashBW = cv2.cvtColor(gwash, cv2.COLOR_BGR2GRAY) #change to grayscale


	# plt.imshow(gwashBW, 'gray') #this is matplotlib solution (Figure 1)
	# plt.xticks([]), plt.yticks([])
	# plt.show()

	cv2.imshow('gwash', gwashBW) #this is for native openCV display
	cv2.waitKey(0)

	ret,thresh1 = cv2.threshold(gwashBW, threshold, 255, cv2.THRESH_BINARY) #the value of 15 is chosen by trial-and-error to produce the best outline of the skull
	kernel = np.ones((kernel_size, kernel_size),np.uint8) #square image kernel used for erosion
	erosion = cv2.erode(thresh1, kernel,iterations = iteration) #refines all edges in the binary image

	opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image

	# plt.imshow(closing, 'gray') #Figure 2
	# plt.xticks([]), plt.yticks([])
	# plt.show()

	contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation

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

	cv2.drawContours(closing, [cnt], 0, (100, 100, 100), 3, maxLevel = maxlevel)
	cv2.imshow('cleaner', closing)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	plt.imshow(closing, 'gray') #Figure 3
	plt.xticks([]), plt.yticks([])
	plt.show()

if __name__ == "__main__":
	# outline("xy4.tif", 5, 1, 5, 0)

	outline("Cells_KB.jpg", threshold = 4, iteration = 1, kernel_size = 5, maxlevel = 0)

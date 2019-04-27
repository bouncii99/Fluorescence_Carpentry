from PIL import Image
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time 

def timer():
    '''
    An ugly timer function.  Do not do this!  I am being lazy
    and programming poorly here for 2 reasons:

        1. I am lazy at times.

        2. I want to illustrate how every function in python is
           actually a class object.  As you can see, here in this case,
           I assign a value (t0) to the timer object, and handle that
           accordingly.

    This timer function needs to be called once, and when called again it
    will print the time elapsed.
    '''
    if not hasattr(timer, 't0'):
        timer.t0 = None
    if timer.t0 is None:
        timer.t0 = time.time()
    else:
        print("%.2f" % (time.time() - timer.t0))
        timer.t0 = None

def binary(image,threshold):


	file = Image.open(image)
	width, height = file.size
	new_image = Image.new('1', (width, height))
	max_intensity = 0
	min_intensity = 65536

	for x in range(width):
		for y in range(height):

			pxl = file.getpixel((x, y))

			if pxl > max_intensity:
				max_intensity = pxl

			if pxl < min_intensity:
				min_intensity = pxl

	print(max_intensity)
	print(min_intensity)

	for x in range(width):
		for y in range(height):

			pxl_2 = file.getpixel((x, y))
			cutoff = (max_intensity - min_intensity) * threshold
			if cutoff != 0:
				if pxl_2 >= cutoff:
					new_image.putpixel((x,y), 1)

				else:
					new_image.putpixel((x,y), 0)

			if cutoff == 0:
				if pxl_2 > cutoff:
					new_image.putpixel((x,y), 1)
				else:
					new_image.putpixel((x,y), 0)


	new_image.save("binary_image.png", "PNG")
	#new_image.show()

	return "binary_image.png"


def denoise(image):

	file = Image.open(image)
	width, height = file.size
	new_image = Image.open(image)
	for x in range(1, width - 1):
		for y in range(1, height - 1):


			pxl_top = file.getpixel((x, y - 1))
			pxl_bottom = file.getpixel((x, y + 1))
			pxl_left = file.getpixel((x - 1, y))
			pxl_right = file.getpixel((x + 1, y))

			neighbor_pxl = pxl_top + pxl_bottom + pxl_left + pxl_right

			if neighbor_pxl == 0:
				new_image.putpixel((x,y), 0)

			if neighbor_pxl == 4:
				new_image.putpixel((x,y), 1)

	new_image.save("denoised_binary_image.png", "PNG")
	#new_image.show()

	return "denoised_binary_image.png"


def hyper_denoise(image):

	direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

	file = Image.open(image)

	width, height = file.size

	new_image = Image.open(image)

	
	# is it suppose to be 3? every 3 or every 2?

	for x in range(1, width - 1):
		for y in range(1, height - 1):

			neighbor_pixel = 0

			for i in direction:

				pxl_3 = file.getpixel((x + i[0], y + i[1]))

				neighbor_pixel = neighbor_pixel + pxl_3

			if neighbor_pixel >= 5:
				new_image.putpixel((x,y), 1)

			else:
				new_image.putpixel((x,y), 0)

	new_image.save("hyper_denoised_binary_image.png", "PNG")
	# new_image.show()

	return "hyper_denoised_binary_image.png"



	# for i in range(iteration):
	# 	di = denoise(image)

def ultra_hyper_denoise(image):

	direction = [(-2,-2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, -2), (
		-1, -1), (-1, 0), (-1, 1), (-1,2),(0,-2),(0, -1), (0, 0), (0, 1), (0,2),(
		1, -2), (1, -1), (1, 0), (1, 1), (1,2),(2,-2),(2, -1), (2, 0), (2,1), (2,2)]

	file = Image.open(image)

	width, height = file.size

	new_image = Image.open(image)

	
	# is it suppose to be 3? every 3 or every 2?

	for x in range(1, width - 1):
		for y in range(1, height - 1):

			neighbor_pixel = 0

			for i in direction:

				pxl_3 = file.getpixel((x + i[0], y + i[1]))

				neighbor_pixel = neighbor_pixel + pxl_3

			if neighbor_pixel >= 13:
				new_image.putpixel((x,y), 1)

			else:
				new_image.putpixel((x,y), 0)

	new_image.save("ultra_hyper_denoised_binary_image.png", "PNG")
	# new_image.show()

	return "ultra_hyper_denoised_binary_image.png"



	# for i in range(iteration):
	# 	di = denoise(image)

def background_corr(image, background_threshold):
	# max_intensity - min_intensity 
	# * 0.5
	# for intensitoy samller than half, find medium in first 25%, then take that out from the lower 50%
	

	file = Image.open(image)
	width, height = file.size


	new_image = Image.open(image)

	max_intensity = 0

	min_intensity = 65536

	for x in range(width):
		for y in range(height):

			pxl = file.getpixel((x, y))

			if pxl > max_intensity:
				max_intensity = pxl

			if pxl < min_intensity:
				min_intensity = pxl

	background = (max_intensity - min_intensity) * background_threshold

	for x in range(width):
		for y in range(height):

			if file.getpixel((x, y)) < background:
				new_image.putpixel((x,y), 0)

	new_image.save("background_corrected.tif")
	# new_image.show()

	return "background_corrected.tif"

# def countour(image):

# 	image = cv2.imread('image')
# 	imgray = cv2.cvtColor(im.cv2.)


def histo_plot(image):
	

	file = Image.open(image)
	width, height = file.size

	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection = '3d')

	xpos = []
	ypos = []
	zpos = []

	print("Compiling data\n----------------")
	timer ()
	for x in range(width):
		for y in range(height):

			# print(x,y)
			
			xpos.append(x)
			ypos.append(y)

			pxl = file.getpixel((x, y))
			zpos.append(pxl)

	num_elements = len(xpos)

	dx = np.ones(num_elements)
	dy = np.ones(num_elements)
	dz = np.ones(num_elements)

	t1 = time.time()


	print("Data compiled: ")
	timer()
	print("\n\n----------------")

	print("Plotting data\n----------------")
	timer()
	ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color= '#00ceaa')

	plt.show()
	plt.save("histo_plot.png")

	print("Plotting completed: ")
	timer()

	return ax1

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

	# cnt is in numpy array, the following code turn it into a list 
	countour_list = []
	for i in cnt:
		countour_list.append((i[0][0], i[0][1]))

	# print(countour_list)

	cv2.drawContours(closing, [cnt], 0, (100, 100, 100), 3, maxLevel = maxlevel)
	cv2.imshow('cleaner', closing)
	cv2.waitKey(0)
	#cv2.destroyAllWindows()

	# The following code generates the countouring image
	file = Image.open(image)
	width, height = file.size
	new_image = Image.new('1', (width, height))

	for x in range(width):
		for y in range(height):
			new_image.putpixel((x,y), 0)

	for i in countour_list:
		new_image.putpixel(i, 1)  


	new_image.show()


if __name__ == "__main__":

	plt = histo_plot("Tiny.jpg")

	# plt.show()
	
	# a = binary("Cells_KB.jpg", 0.01)

	# b = denoise(a)

	# c = hyper_denoise(a)

	# outline(c, threshold = 1, iteration = 1, kernel_size = 3, maxlevel = 0)

	# file = Image.open("n1001z3c2.tif")
	# file.show()

	# d = background_corr("n1001z3c2.tif", 0.25)




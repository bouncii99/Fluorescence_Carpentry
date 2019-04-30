from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def binary(image, threshold):
	'''
	This function converts any form of image to a binary image.
	This is done so that the image can be scanned and denoised
	more efficiently.
	**Parameters**
		image: *image*
			This is the image that has to be converted to a binary image.
		threshold: *float*
			A % of the minimum pixel intensity for it to be considered
			white or black.
	**Returns**
		binary_image: *image, png*
			This is a binary image. i.e. It has only 2 possible values
			for it's pixels, 1 or 0.
	'''
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


	for x in range(width):
		for y in range(height):

			pxl_2 = file.getpixel((x, y))
			cutoff = (max_intensity - min_intensity) * threshold
			if cutoff != 0:
				if pxl_2 >= cutoff:
					new_image.putpixel((x, y), 1)

				else:
					new_image.putpixel((x, y), 0)

			if cutoff == 0:
				if pxl_2 > cutoff:
					new_image.putpixel((x, y), 1)
				else:
					new_image.putpixel((x, y), 0)

	new_image.save("binary_image.png", "PNG")

	return "binary_image.png"


def denoise(image):
	'''
	This function takes the resulting binary image and reduces the background
	noise. It does this by totaling the pixel value surrounding the current
	pixel and comparing it's value to the max possible sum, 4.
	**Parameters**
		image: *binary, png*
	**Returns**
		denoised_image: *binary, png*
	'''
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
				new_image.putpixel((x, y), 0)

			if neighbor_pxl == 4:
				new_image.putpixel((x, y), 1)

	new_image.save("denoised_binary_image.png", "PNG")

	return "denoised_binary_image.png"


def hyper_denoise(image):
	'''
	Further filtering the binary denoised image by parsing it
	through one more denoising function. This function uses more
	stringent conditions to reduce noise and filters out noise by
	changing their pixel values to 0.
	'''
	direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
				 (0, 1), (1, -1), (1, 0), (1, 1)]

	file = Image.open(image)
	width, height = file.size
	new_image = Image.open(image)

	for x in range(1, width - 1):
		for y in range(1, height - 1):

			neighbor_pixel = 0

			for i in direction:

				pxl_3 = file.getpixel((x + i[0], y + i[1]))

				neighbor_pixel = neighbor_pixel + pxl_3

			if neighbor_pixel >= 5:
				new_image.putpixel((x, y), 1)

			else:
				new_image.putpixel((x, y), 0)

	new_image.save("hyper_denoised_binary_image.png", "PNG")
	# new_image.show()

	return "hyper_denoised_binary_image.png"


def ultra_hyper_denoise(image):
	'''
	Further filtering the binary denoised image by parsing it
	through a third denoising function. As we move along this
	iterative path of reducing noise, the filters get progressively
	more stringent with their filtering thresholds. Not used currently.
	'''

	direction = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, -2),
				 (-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -2), (0, -1),
				 (0, 0), (0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1),
				 (1, 2), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]

	file = Image.open(image)
	width, height = file.size
	new_image = Image.open(image)

	for x in range(1, width - 1):
		for y in range(1, height - 1):

			neighbor_pixel = 0

			for i in direction:

				pxl_3 = file.getpixel((x + i[0], y + i[1]))

				neighbor_pixel = neighbor_pixel + pxl_3

			if neighbor_pixel >= 13:
				new_image.putpixel((x, y), 1)

			else:
				new_image.putpixel((x, y), 0)

	new_image.save("ultra_hyper_denoised_binary_image.png", "PNG")
	# new_image.show()

	return "ultra_hyper_denoised_binary_image.png"


def outline(image, threshold, iteration, kernel_size, maxlevel):
	'''
	Given a filtered binary image, this function will return a numpy
	array of the largest contour on in the image. This function hence
	returns the coordinates of the cell boundary.
	**Parameters**
		image: *binary, png*
			Takes in a binary, denoised image.
		threshold: *float*
			A % of the minimum pixel intensity for it to be considered
			white or black.
		iteration: *int*
			Number of times we want to erode the outer pixel layer of
			the image.
		kernel_size: *odd numbered matrix, int*
			A matrix that dictates the number of neighbouring cells'
			pixel values be scanned in order to decide whether the
			current pixel value is 1 or 0.
		maxlevel: *int*
			It is the maximum level for drawn contours. Given 0 as we
			only want the largest contour.
	**Returns**
		width: *int*
			Width of image
		height: *int*
			Height of image
		cnt: *numpy.ndarray*
			Numpy array containing the coordinates and grayscale
			value of the largest contour in the image.
	'''

	# Convert denoised image to binary image (black & white) and show.
	# Press any key to proceed
	imported_img = cv2.imread(image)
	img_BW = cv2.cvtColor(imported_img, cv2.COLOR_BGR2GRAY)
	cv2.imshow('imported_img', imported_img)
	cv2.waitKey(0)

	# '15' is chosen by trial-and-error to produce the best outline of the cell
	ret, thresh1 = cv2.threshold(img_BW, threshold, 255, cv2.THRESH_BINARY)

	# square image kernel used for erosion
	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	# refines all edges in the binary image
	erosion = cv2.erode(thresh1, kernel, iterations=iteration)

	# this is for further removing small noises and holes in the image
	opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

	# Image dimension:
	width, height = len(closing[0]), len(closing)

	# This finds contours with no approximation. The resulting list of
	# contours contain all points on object boundary.

	# For Python 2, use the following line:
	contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# For Python 3, use the following line:
	# img, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE,
	#                                             cv2.CHAIN_APPROX_NONE)

	# Clean up the cell boundary. Press any key to proceed
	cv2.imshow('Clean: Press any key to proceed', closing)
	cv2.drawContours(closing, contours, -1, (100, 100, 100), 4)
	cv2.waitKey(0)

	# List, holds all area coordinates
	areas = []

	for contour in contours:
		ar = cv2.contourArea(contour)
		areas.append(ar)

	# Finding index of list elements with largest area
	max_area = max(areas)
	max_area_index = areas.index(max_area)

	# Numpy array, contains coordinates of largest area
	cnt = contours[max_area_index]

	# Further smooth out the cell boundary. Press any key to proceed
	cv2.drawContours(closing, [cnt], 0, (100, 100, 100), 3, maxLevel=maxlevel)
	cv2.imshow('Cleaner: Press any key to proceed', closing)
	cv2.waitKey(0)

	return width, height, cnt


def edge(width, height, cnt, output):
	'''
	This function takes the smoothened out contours obtained in
	*outline* function and process it to create a list of coordinates
	for the boundary of the main cell body (object with largest area)
	and its image.
	**Parameters**
		width, height: *int*
			Dimension of image
		cnt: *numpy.ndarray*
			Numpy array containing the coordinates and grayscale value
			of the largest contour in the image.
	**Returns**
		contour_list: *list, tuples*
			List containing the coordinates (tuples) of the cell boundary
		image: *binary, png*
			This image contains only the cell boundary.
	'''

	# Converting numpy array into a list of tuples
	countour_list = []
	for i in cnt:
		countour_list.append((i[0][0], i[0][1]))

	# print(countour_list)

	# The following code generates the image of the cell edge
	new_image = Image.new('1', (width, height))

	for x in range(width):
		for y in range(height):
			new_image.putpixel((x, y), 0)

	for i in countour_list:
		new_image.putpixel(i, 1)

	new_image.save(output)

	return countour_list


def centroid(cnt, image2annotate, output):

	'''
	This section was referenced to the help source:
	https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
	This function finds the centroid of a cell based on the cell
	boundary found from *edge* function.
	**Parameters**
		cnt: *numpy.ndarray*
			Numpy array containing the coordinates and grayscale value
			of the largest contour in the image.
		image2annotate: *binary, tif*
			An image with the cell boundary used to annotate where the
			centroid is.
	**Returns**
		centroid: *tuple*
			coordinate of the centroid of the cell
		Image: *.tif*
			Illustrative image with the centroid
	'''

	# Convert image of cell boundary to numpy array for cv2
	image_to_annotate = cv2.imread(image2annotate)

	# calculate moments of binary image
	M = cv2.moments(cnt)
	# print M

	# calculate x,y coordinate of centroid
	cX = int(round(M["m10"] / M["m00"]))
	cY = int(round(M["m01"] / M["m00"]))
	centroid = tuple((cX, cY))

	# put text and highlight the centroid
	cv2.circle(image_to_annotate, centroid, 3, (225, 225, 225), -1)
	cv2.putText(
		image_to_annotate, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 255), 1)

	# Save the image. Uncomment the last 2 codes to show:
	cv2.imwrite(output, image_to_annotate)
	cv2.imshow("centroid", image_to_annotate)
	cv2.waitKey(0)

	return centroid


def shapes():
	'''
	This functions defines the sizes of the shapes to be drawn
	with known centres, outlines and values. Enables the unit test
	function to run, checking the accuracy of the code.

	**Returns**
		Images: 2 separate grayscale images of a square and an ellipse
	'''

	# Draw and fill a square and an ellipse with cv2 built-in functions:
	square = np.zeros((512, 512, 3), np.uint8)
	cv2.rectangle(square, (1, 1), (51, 51), (0, 255, 0), -1)

	ellipse = np.zeros((512, 512, 3), np.uint8)
	cv2.ellipse(ellipse, (256, 256), (100, 50), 0, 0, 360, (0, 255, 0), -1)

	# Save shape images as .jpg files. Then read them with with cv2 to 
	# convert to grayscale image and overwrite the original files into 
	# grayscale files
	cv2.imwrite("square.jpg", square)
	cv2.imwrite("ellipse.jpg", ellipse)

	gray = cv2.imread("square.jpg", 0)
	cv2.imwrite("square.jpg", gray)

	gray = cv2.imread("ellipse.jpg", 0)
	cv2.imwrite("ellipse.jpg", gray)

	# cv2.imshow("square", square)
	# cv2.waitKey()
	# cv2.imshow("ellipse", ellipse)
	# cv2.waitKey()


def unit_test(filename):
	'''
	Unit test function in order to test out the code that will
	run the current code on predefined functions and known shapes
	and asserts it against known answers. If the answers match,
	the function will run, else display an error.

	**Parameters**
		filename: *str* + '.jpg'
			Appropriate filename of shape images generated by *shapes()* function
	'''

	# Test image is clean and already binary so there is no need for denoising
	w, h, cnt = outline(filename, 1, 1, 3, 0)
	
	# Square Centroid  = 26,26
	# Ellipse Centroid  = 256, 256
	centroid_dict = {'square': (26,26), 'ellipse': (256,256)}

	cnt_output = filename.split('.jpg')[0] + '_contour.jpg'
	edge(w, h, cnt, output=cnt_output)

	cent_output = filename.split('.jpg')[0] + '_centroid.jpg'
	cent = centroid(cnt, image2annotate=cnt_output, output=cent_output)
	print(cent)
	assert cent == centroid_dict[filename.split('.jpg')[0]],'Centroid incorrect'


'''
********************************************************************
The following code is for project continuation. Currently not in use
********************************************************************
'''


def timer():

	'''
	This timer function was written by Henry Herbol.
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


def background_corr(image, background_threshold):

	'''
	This function was used as another approach to denoise the image.
	Here to check robustness of code. Not used currently.
	'''
	# max_intensity - min_intensity
	# * 0.5
	# for intensitoy samller than half,
	# find medium in first 25%, then take that out from the lower 50%

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
				new_image.putpixel((x, y), 0)

	new_image.save("background_corrected.tif")

	return "background_corrected.tif"


def histo_plot(image):
	'''
	An extension of backgroud_corr function that plots the image
	after corrections. Not used currently, here to check
	robustness of code.
	'''
	file = Image.open(image)
	width, height = file.size
	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')

	xpos = []
	ypos = []
	zpos = []

	print("Compiling data\n----------------")
	timer()
	for x in range(width):
		for y in range(height):
			xpos.append(x)
			ypos.append(y)

			pxl = file.getpixel((x, y))
			zpos.append(pxl)

	num_elements = len(xpos)

	dx = np.ones(num_elements)
	dy = np.ones(num_elements)
	dz = np.ones(num_elements)

	print("Data compiled: ")
	timer()
	print("\n\n----------------")

	print("Plotting data\n----------------")
	timer()
	ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')

	plt.show()
	plt.save("histo_plot.png")

	print("Plotting completed: ")
	timer()

	return ax1


if __name__ == "__main__":

	filename = 'Cells_KB.jpg'
	a = binary(filename, 0.01)

	c = hyper_denoise(a)

	w, h, cnt = outline(c, 1, 1, 3, 0)

	filename1 = filename.split('.')[0] + "_contour.jpg"
	edge(w, h, cnt, output = filename1)

	filename2 = filename.split('.')[0] + '_centroid.jpg'
	centroid(cnt, image2annotate = filename1,
	      output = filename2)
	
	################################################################# 
	# Initializing unit test for predetermined shapes here. For square
	# use 'square.jpg'. For ellipse, use 'ellipse.jpg'
	#################################################################
	# shapes() 
	# unit_test('ellipse.jpg')
	# unit_test('square.jpg')
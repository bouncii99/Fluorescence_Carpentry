from PIL import Image
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')

# xpos = [1,2,3,4,5,6,7,8,9,10]
# ypos = [2,3,4,5,1,6,2,1,7,2]
# num_elements = len(xpos)
# zpos = [0,0,0,0,0,0,0,0,0,0]
# dx = np.ones(10)
# dy = np.ones(10)
# dz = [1,2,3,4,5,6,7,8,9,10]

# ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
# plt.show()

# file = Image.open("Cells_KB.jpg")
# width, height = file.size

# print(width, height)

# def histo_plot(image):
	

# 	file = Image.open(image)
# 	width, height = file.size

# 	fig = plt.figure()
# 	ax1 = fig.add_subplot(111, projection = '3d')

# 	xpos = []
# 	ypos = []
# 	zpos = []

# 	num_elements = width * height
# 	dx = np.ones(num_elements)
# 	dy = np.ones(num_elements)
# 	dz = np.ones(num_elements)

# 	for x in range(width):
# 		for y in range(height):

# 			# print(x,y)
			
# 			# xpos.append(x)
# 			# ypos.append(y)

			
# 			pxl = file.getpixel((x, y))
# 			zpos.append(pxl)
# 			ax1.bar3d(x, y, pxl, dx, dy, dz, color= '#00ceaa')

# 			plt.show()
# 			# ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color= '#00ceaa')

# 	plt.show()


# 	return ax1



def histo_plt(image):

	gwash = cv2.imread(image) #import image
	gwashBW = cv2.cvtColor(gwash, cv2.COLOR_BGR2GRAY)

	file = Image.open(image)
	width, height = file.size

	# setup the figure and axes
	fig = plt.figure(figsize=(width, height))
	ax1 = fig.add_subplot(121, projection='3d')

	xpos = []
	ypos = []
	zpos = []


	for x in range(width):
			for y in range(height):

				print(x,y)
				
				xpos.append(x)
				ypos.append(y)

				pxl = file.getpixel((x, y))
				zpos.append(pxl)

	num_elements = len(xpos)

	dx = np.ones(num_elements)
	dy = np.ones(num_elements)
	dz = np.ones(num_elements)


	# ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
	# ax1.set_title('Shaded')


	# plt.show()

	xpos = np.array(xpos)
	ypos = np.array(ypos)
	zpos = np.array(zpos)

	N = int(len(zpos)**.1)
	z = zpos.reshape(N, N)
	plt.imshow(z, extent=(np.amin(xpos), np.amax(xpos), np.amin(ypos), np.amax(ypos)), norm=LogNorm(), aspect = 'auto')
	plt.colorbar()
	plt.show()


if __name__ == "__main__":
	histo_plt("Tiny.tif")
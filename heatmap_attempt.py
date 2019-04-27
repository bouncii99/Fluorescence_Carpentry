from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from scipy.interpolate import interp2d

# f will be a function with two arguments (x and y coordinates),
# but those can be array_like structures too, in which case the
# result will be a matrix representing the values in the grid 
# specified by those arguments

file = Image.open("Tiny.tif")
width, height = file.size

x_list = []
y_list = []
z_list = []


for x in range(width):
		for y in range(height):

			# print(x,y)
			
			x_list.append(x)
			y_list.append(y)

			pxl = file.getpixel((x, y))
			z_list.append(pxl)

n = width

z_list_final = [z_list[i * n:(i + 1) * n] for i in range((len(z_list) + n - 1) // n )]  

# print z_list_final

# x_list = np.array([

# 	-1,2,10,3])
# y_list = np.array([3,-3,4,7])
# z_list = np.array([5,1,2.5,4.5])





f = interp2d(x_list, y_list, z_list, kind="linear")

x_coords = np.arange(min(x_list),max(x_list)+1)
y_coords = np.arange(min(y_list),max(y_list)+1)
Z = f(x_coords,y_coords)

fig = plt.imshow(Z,
           extent=[min(x_list),max(x_list),min(y_list),max(y_list)],
           origin="lower")

plt.colorbar()

# Show the positions of the sample points, just to have some reference
fig.axes.set_autoscale_on(False)
plt.scatter(x_list,y_list,400,facecolors='none')


plt.show()
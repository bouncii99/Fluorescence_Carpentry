from PIL import Image

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

	new_image.show()

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
	new_image.show()

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
	new_image.show()

	return "hyper_denoised_binary_image.png"



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
	new_image.show()

	return "background_corrected.tif"



if __name__ == "__main__":
	
	# a = binary("n1001z3c2.tif", 0.01)

	# b = denoise(a)

	# c = hyper_denoise(a)

	file = Image.open("xy4.tif")
	file.show()

	d = background_corr("xy4.tif", 0.25)
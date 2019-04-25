from PIL import Image



def binary(image,threshold):


	file = Image.open(image)
	width, height = file.size


	new_image = Image.new('1', (width, height))


	binary_pixel = {}

	for x in range(width):
		for y in range(height):

			pxl = file.getpixel((x, y))

			if pxl > 0:
				new_image.putpixel((x,y), 1)

			if pxl == 0:
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


def hyper_denoise(image, iteration):

	direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

	file = Image.open(image)

	width, height = file.size

	new_image = Image.open(image)

	if 

	for x in range(1, width - 1):
		for y in range(1, height - 1):

			for i in directions:



	pass
	# for i in range(iteration):
	# 	di = denoise(image)





if __name__ == "__main__":
	
	a = binary("Cells_KB.jpg", 0.5)

	b = denoise(a)

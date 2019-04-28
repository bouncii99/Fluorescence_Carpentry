from PIL import Image


def image_sett(self, image):
    self.file = Image.open(image)
    self.width, self.height = file.size
    return self.file, self.width, self.height


def binary(image, threshold):
    '''
    This function converts any form of image to a binary image.
    This is done so that the image can be scanned and denoised
    more efficiently.

    **Parameters**

        image: *image*
            This is the image that has to be converted to a binary image.

        threshold: *float*
            A % of the minimum pixel intensity for it to be considered white
            or black.

    **Returns**

        binary_image: *image, png*
            This is a binary image. i.e. It has only 2 possible values for it's
            pixels, 1 or 0.

    '''
    file = Image.open(image)
    width, height = file.size
    new_image = Image.new('1', (self.width, self.height))
    max_intensity = 0
    min_intensity = 65536

    for x in range(self.width):
        for y in range(self.height):

            pxl = file.getpixel((x, y))

            if pxl > max_intensity:
                max_intensity = pxl

            if pxl < min_intensity:
                min_intensity = pxl

    print max_intensity, min_intensity

    for x in range(self.width):
        for y in range(self.height):

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
    new_image.show()
    return "binary_image.png"


if __name__ == "__main__":
    a = binary("Cells_KB.jpg", 0.001)

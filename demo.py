from PIL import Image


def binary(image, threshold):

    file = Image.open(image)
    width, height = file.size
    new_image = Image.new('1', (width, height))
    max_intensity = 0
    min_intensity = 65536
    print type(threshold)

    for x in range(width):
        for y in range(height):

            pxl = file.getpixel((x, y))

            if pxl > max_intensity:
                max_intensity = pxl

            if pxl < min_intensity:
                min_intensity = pxl

    return(max_intensity, min_intensity)

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
    new_image.show()
    return "binary_image.png"


if __name__ == "__main__":
    a = binary("Cells_KB.jpg", 0.01)

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


class cell_outline(object):
    '''
    Cell outline class represents the first milestone of the project. 
    The function of this class is to return the perimeter of the largest
    cell in a given image. 
    '''

    def init(self, img):
        self.fptr = Image.open(img)

    def image_settings(self):
        self.width, self.height = self.fptr.size
        empty_image = Image.new('1', (width, height))
        self.max_intensity = 0
        self.min_intensity = 65536
    

    def binary(self, img, threshold):
        for x in range(self.width):
                for y in range(self.height):
                    pxl = self.fptr.getpixel((x, y))
                    if pxl > max_intensity:
                        max_intensity = pxl
                    if pxl < min_intensity:
                        min_intensity = pxl
                return self.max_intensity, self.min_intensity

        for x in range(self.width):
            for y in range(self.height):
                pxl_2 = self.fptr.getpixel((x, y))
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


if __name__ == "__main__":
    cell_outline("Cells_KB.jpg")
    

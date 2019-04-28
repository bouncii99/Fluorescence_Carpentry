import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image, ImageEnhance

img = cv.imread('cell_confocal.jpg', 1)
# cv2.imshow('Color Image', img)
plt.imshow(img, cmap = 'gray')
plt.show()
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # Removes xticks and yticks
plt.show()
img1 = cv.imread('cells_KB.jpg', 0)
# cv2.imshow('Grayscale Image', img1)


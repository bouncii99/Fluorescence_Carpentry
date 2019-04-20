import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance


# def noisered_laplace(image):
#     laplacian_64f = cv2.Laplacian(image, cv2.CV_64F)
#     abs_laplacian = np.absolute(laplacian_64f)
#     image = np.uint8(abs_laplacian)
# img2 = noisered_laplace(img)
# plt.imshow(img2)
# img1 = cv2.Laplacian(img, cv2.CV_64F)
# plt.imshow(img1)


img = cv2.imread("cells_KB.jpg")
laplacian_64f = cv2.Laplacian(img,cv2.CV_64F)
abs_laplacian = np.absolute(laplacian_64f)
image = np.uint8(abs_laplacian)
plt.imshow(abs_laplacian,cmap = 'gray')
plt.title('Abs_Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()
= 'gray')
plt.imshow(laplacian_64f, cmap= 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()
# img1 = ImageEnhance.Contrast(img).enhance(2.5)
# img1.show()
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
# plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.show()

# edges = cv2.Canny(laplacian,100,200)
# plt.subplot(121),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

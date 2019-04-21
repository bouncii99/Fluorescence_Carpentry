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

W, H = len(abs_laplacian[0]), len(abs_laplacian)
laplacian = Image.new("RGB", (W, H), color=(0, 0, 0))

for x in range(W):
	for y in range(H):
		if np.any(image[y][x] != 0):
			laplacian.putpixel((x,y), (255,255,255))

laplacian.save('laplacian.png')

# plt.imshow(abs_laplacian,cmap = 'gray')
# plt.title('Abs_Laplacian'), plt.xticks([]), plt.yticks([])
# plt.show()
# cv2.imwrite('abs_laplacian.png', abs_laplacian)

# plt.imshow(laplacian_64f, cmap= 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.show()
# img1 = ImageEnhance.Contrast(img).enhance(2.5)
# img1.show()
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
# plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.show()

# edges = cv2.Canny(laplacian,100,200)
# plt.subplot(121),plt.imshow(laplacian_64f, cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

if __name__ == "__main__":
	pass

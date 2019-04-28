import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150) #sick
    return canny

sourceimage = cv2.imread('Cells_KB.jpg')
img = np.copy(sourceimage)
canny = canny(img)
cv2.imshow("result", canny)
cv2.waitKey(0)
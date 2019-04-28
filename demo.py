import cv2
from PIL import Image
import numpy as np

square = np.zeros((512,512,3), np.uint8)
cv2.rectangle(square,(0,0),(50,50),(0,255,0),-1)
ellipse = np.zeros((512,512,3), np.uint8)
cv2.ellipse(ellipse,(256,256),(100,50),0,0,360,(0,255,0),-1)
cv2.imwrite("square.jpg", square)
cv2.imwrite("ellipse.jpg", ellipse)
cv2.imshow("square", square)
cv2.waitKey()
cv2.imshow("ellipse", ellipse)
cv2.waitKey()
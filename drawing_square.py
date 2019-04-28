import numpy as np
import cv2 
from PIL import Image
 

# Create a black image
img = np.zeros((512,512,3), np.uint8) 
# Draw a diagonal blue line with thickness of 5 px
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)


cv2.imwrite('color_img.jpg', img)
cv2.imshow("image", img)
cv2.waitKey()
'''
The final aim of this python script is to be able to run through a binary image and detect edges. 
When it detects edges, the coordinates of the edge will be saved in a pre-existing list. 
'''
import time
import random as R
import numpy as np
from PIL import Image

def get_img(filename):
    if ".png" in filename:
        filename = filename.split(".png")[0]
    elif ".jpg" in filename:
        filename = filename.split(".jpg")[0]
    img = Image.open(filename + ".png")

def img_size(img):
    width, height = img.size
    return width, height

def pos_chk(x, y, width, height):
    return x >= 0 and x < width and y >= 0 and y < height

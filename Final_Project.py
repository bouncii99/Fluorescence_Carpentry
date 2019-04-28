from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def timer():
    '''
    This timer function was written by Henry Herbol.
    This timer function needs to be called once, and when called again it
    will print the time elapsed.
    '''
    if not hasattr(timer, 't0'):
        timer.t0 = None
    if timer.t0 is None:
        timer.t0 = time.time()
    else:
        print("%.2f" % (time.time() - timer.t0))
        timer.t0 = None


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
    new_image = Image.new('1', (width, height))
    max_intensity = 0
    min_intensity = 65536

    for x in range(width):
        for y in range(height):

            pxl = file.getpixel((x, y))

            if pxl > max_intensity:
                max_intensity = pxl

            if pxl < min_intensity:
                min_intensity = pxl

    print max_intensity, min_intensity

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

    return "binary_image.png"


def denoise(image):
    '''
    This function takes the resulting binary image and reduces the background
    noise. It does this by totaling the pixel value surrounding the current
    pixel and comparing it's value to the max possible sum, 4.

    **Parameters**
        image: *binary, png*

    **Returns**
        denoised_image: *binary, png*
    '''
    file = Image.open(image)
    width, height = file.size
    new_image = Image.open(image)
    for x in range(1, width - 1):
        for y in range(1, height - 1):

            pxl_top = file.getpixel((x, y - 1))
            pxl_bottom = file.getpixel((x, y + 1))
            pxl_left = file.getpixel((x - 1, y))
            pxl_right = file.getpixel((x + 1, y))

            neighbor_pxl = pxl_top + pxl_bottom + pxl_left + pxl_right

            if neighbor_pxl == 0:
                new_image.putpixel((x, y), 0)

            if neighbor_pxl == 4:
                new_image.putpixel((x, y), 1)

    new_image.save("denoised_binary_image.png", "PNG")

    return "denoised_binary_image.png"


def hyper_denoise(image):
    '''
    Further filtering the binary denoised image by parsing it
    through one more denoising function. This function uses more
    stringent conditions to reduce noise and filters out noise by
    changing their pixel values to 0.
    '''
    direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                 (0, 1), (1, -1), (1, 0), (1, 1)]

    file = Image.open(image)
    width, height = file.size
    new_image = Image.open(image)

    for x in range(1, width - 1):
        for y in range(1, height - 1):

            neighbor_pixel = 0

            for i in direction:

                pxl_3 = file.getpixel((x + i[0], y + i[1]))

                neighbor_pixel = neighbor_pixel + pxl_3

            if neighbor_pixel >= 5:
                new_image.putpixel((x, y), 1)

            else:
                new_image.putpixel((x, y), 0)

    new_image.save("hyper_denoised_binary_image.png", "PNG")
    # new_image.show()

    return "hyper_denoised_binary_image.png"


def ultra_hyper_denoise(image):
    '''
    Further filtering the binary denoised image by parsing it
    through a third denoising function. As we move along this
    iterative path of reducing noise, the filters get progressively
    more stringent with their filtering thresholds. Not used currently.
    '''

    direction = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, -2),
                 (-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -2), (0, -1),
                 (0, 0), (0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1),
                 (1, 2), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]

    file = Image.open(image)
    width, height = file.size
    new_image = Image.open(image)

    for x in range(1, width - 1):
        for y in range(1, height - 1):

            neighbor_pixel = 0

            for i in direction:

                pxl_3 = file.getpixel((x + i[0], y + i[1]))

                neighbor_pixel = neighbor_pixel + pxl_3

            if neighbor_pixel >= 13:
                new_image.putpixel((x, y), 1)

            else:
                new_image.putpixel((x, y), 0)

    new_image.save("ultra_hyper_denoised_binary_image.png", "PNG")
    # new_image.show()

    return "ultra_hyper_denoised_binary_image.png"


def background_corr(image, background_threshold):

    '''
    This function was used as another approach to denoise the image.
    Here to check robustness of code. Not used currently.
    '''
    # max_intensity - min_intensity
    # * 0.5
    # for intensitoy samller than half,
    # find medium in first 25%, then take that out from the lower 50%

    file = Image.open(image)
    width, height = file.size
    new_image = Image.open(image)
    max_intensity = 0
    min_intensity = 65536

    for x in range(width):
        for y in range(height):

            pxl = file.getpixel((x, y))

            if pxl > max_intensity:
                max_intensity = pxl

            if pxl < min_intensity:
                min_intensity = pxl

    background = (max_intensity - min_intensity) * background_threshold

    for x in range(width):
        for y in range(height):

            if file.getpixel((x, y)) < background:
                new_image.putpixel((x, y), 0)

    new_image.save("background_corrected.tif")

    return "background_corrected.tif"


def histo_plot(image):
    '''
    An extension of backgroud_corr function that plots the image
    after corrections. Not used currently, here to check
    robustness of code.
    '''
    file = Image.open(image)
    width, height = file.size 
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    xpos = []
    ypos = []
    zpos = []

    print("Compiling data\n----------------")
    timer()
    for x in range(width):
        for y in range(height):
            xpos.append(x)
            ypos.append(y)

            pxl = file.getpixel((x, y))
            zpos.append(pxl)

    num_elements = len(xpos)

    dx = np.ones(num_elements)
    dy = np.ones(num_elements)
    dz = np.ones(num_elements)

    print("Data compiled: ")
    timer()
    print("\n\n----------------")

    print("Plotting data\n----------------")
    timer()
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')

    plt.show()
    plt.save("histo_plot.png")

    print("Plotting completed: ")
    timer()

    return ax1


def outline(image, threshold, iteration, kernel_size, maxlevel):
    '''
    Given a filtered binary image, this function will return a numpy array
    of the largest contour on in the image. This function hence returns the
    coordinates of the cell boundary.

    **Parameters**
        image: *binary, png*
            Takes in a binary, denoised image.
        threshold: *float*
            A % of the minimum pixel intensity for it to be considered white
            or black.
        iteration: *int*
            Number of times we want to erode the outer pixel layer of the
            image.
        kernel_size: *odd numbered matrix, int*
            A matrix that dictates the number of neighbouring cells' pixel
            values be scanned in order to decide whether the current pixel
            value is 1 or 0.
        maxlevel: *int*
            It is the maximum level for drawn contours. Given 0 as we only
            want the largest contour.

    **Returns**
        contour_list: *list, tuple*
            List containing the coordinates of the largest contour in
            the image.
        image: *binary, png*
            This image contains only the cell boundary.
    '''
    imported_img = cv2.imread(image)
    img_BW = cv2.cvtColor(imported_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('imported_img', imported_img)
    cv2.waitKey(0)

    # '15' is chosen by trial-and-error to produce the best outline of the cell
    ret, thresh1 = cv2.threshold(img_BW, threshold, 255, cv2.THRESH_BINARY)
    # square image kernel used for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # refines all edges in the binary image
    erosion = cv2.erode(thresh1, kernel, iterations=iteration)
    # this is for further removing small noises and holes in the image
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    
    # This finds contours with simple approximation

    # For Python 2, use the following line:
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # # For Python 3, use the following line:
    # img, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    cv2.imshow('cleaner', closing)
    cv2.drawContours(closing, contours, -1, (100, 100, 100), 4)
    cv2.waitKey(0)

    # List, holds all area coordinates
    areas = []

    for contour in contours:
        ar = cv2.contourArea(contour)
        areas.append(ar)

    # Finding index of list elements with largest area
    max_area = max(areas)
    max_area_index = areas.index(max_area)

    # Numpy array, contains coordinates of largest area
    cnt = contours[max_area_index]

    # Converting numpy array into a list of tuples
    countour_list = []
    for i in cnt:
        countour_list.append((i[0][0], i[0][1]))

    # print(countour_list)
    cv2.drawContours(closing, [cnt], 0, (100, 100, 100), 3, maxLevel=maxlevel)
    cv2.imshow('cleaner', closing)
    cv2.waitKey(0)

    # The following code generates the countouring image
    file = Image.open(image)
    width, height = file.size
    new_image = Image.new('1', (width, height))

    for x in range(width):
        for y in range(height):
            new_image.putpixel((x, y), 0)

    for i in countour_list:
        new_image.putpixel(i, 1)

    new_image.show()



def centroid(image, threshold, iteration, kernel_size, image2annotate):

    '''
    This section was referenced to the help source:
    https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

    '''

    gwash = cv2.imread(image) #import image
    gwashBW = cv2.cvtColor(gwash, cv2.COLOR_BGR2GRAY) #change to grayscale

    # image_to_annotate = cv2.imread(image2annotate)

    image_to_annotate = Image.open(image2annotate).convert('RGB')

    image_to_annotate.save("image_to_annotate.tif")

    image_to_annotate = cv2.imread("image_to_annotate.tif")


    # image_to_annotateBW = cv2.imread(image2annotate)
    # image_to_annotate = cv2.cvtColor(image_to_annotateBW,cv2.COLOR_GRAY2RGB)



    # plt.imshow(gwashBW, 'gray') #this is matplotlib solution (Figure 1)
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    cv2.imshow('gwash', gwashBW) #this is for native openCV display
    cv2.waitKey(0)

    ret,thresh1 = cv2.threshold(gwashBW, threshold, 255, cv2.THRESH_BINARY) #the value of 15 is chosen by trial-and-error to produce the best outline of the skull
    kernel = np.ones((kernel_size, kernel_size),np.uint8) #square image kernel used for erosion
    erosion = cv2.erode(thresh1, kernel,iterations = iteration) #refines all edges in the binary image

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image

    # plt.imshow(closing, 'gray') #Figure 2
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #find contours with simple approximation

    cv2.imshow('cleaner', closing) #Figure 3
    cv2.drawContours(closing, contours, -1, (100, 100, 100), 4)
    cv2.waitKey(0)

    areas = [] #list to hold all areas

    for contour in contours:
        ar = cv2.contourArea(contour)
        areas.append(ar)

    max_area = max(areas)
    max_area_index = areas.index(max_area) #index of the list element with largest area

    cnt = contours[max_area_index] #largest area contour
    
    # calculate moments of binary image
    M = cv2.moments(cnt)
    # print M
    
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    

    # put text and highlight the center

    cv2.circle(image_to_annotate, (cX, cY), 3, (225, 225, 225), -1)
    cv2.putText(image_to_annotate, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 255), 1)

    # cv2.circle(gwash, (cX, cY), 5, (120, 20, 60), -1)
    # cv2.putText(gwash, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 20, 60), 2)

    # display the image
    cv2.imshow("Image", image_to_annotate)
    cv2.waitKey(0)


    return tuple((cX, cY))


if __name__ == "__main__":

    # plt = histo_plot("Tiny.tif")

    # plt.show()


    # a = binary("sc005z3c2.tif", 0.1)

    a = binary("Cells_KB.jpg", 0.01)

    # a = binary("xy4.tif", 0.25)
    # a = binary("n1001z3c2.tif", 0.01)

    # b = denoise(a)

    c = hyper_denoise(a)

    outline(c, threshold = 1, iteration = 1, kernel_size = 3, maxlevel = 0)

    centroid(c, threshold = 1, iteration = 1, kernel_size = 3, image2annotate = "contour.tif")

    # d = background_corr("n1001z3c2.tif", 0.25)


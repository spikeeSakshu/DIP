import cv2
import matplotlib.pyplot as plt
import numpy as np

def edge(img):
    #Use median filter to reduce noise
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    
    #Use adaptive thresholding to create an edge mask
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
       cv2.ADAPTIVE_THRESH_MEAN_C,
       cv2.THRESH_BINARY,
       blockSize=17,
       C=5)
    return img_edge

def Cartoonize(img_rgb):
    num_down=1
    num_bilateral = 5 # number of bilateral filtering steps

    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=4)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    dim = (img_edge.shape[1],img_edge.shape[0])
    img_color = cv2.resize(img_color,dim)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    return img_cartoon

def sharp(img):
    kernel_sharpening = np.array([[0,-1,0], 
                              [-1, 5,-1],
                              [0,-1, 0]])
    
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    return sharpened

def blur(img):
    kernel_3x3 = np.ones((7, 7), np.float32) / 49
    # We apply the filter and display the image
    blurred = cv2.filter2D(img, -1, kernel_3x3)
    return blurred



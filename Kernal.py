import cv2
import matplotlib.pyplot as plt
import numpy as np

def EdgeDetection(img,b,c):
    #Use median filter to reduce noise
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    
    #Use adaptive thresholding to create an edge mask
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
       cv2.ADAPTIVE_THRESH_MEAN_C,
       cv2.THRESH_BINARY,
       blockSize=b,
       C=c)
    return img_edge

def Cartoonize(img_rgb):
    num_bilateral = 5 # number of bilateral filtering steps
    
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=9, sigmaSpace=7)
    
    
    img_edge = EdgeDetection(img_rgb,11,4)
    
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    
    # Combine color image with edge mask & display picture
    # convert back to color, bit-AND with color image
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    
    return img_cartoon

def Sharpen(img):
    kernel_sharpening = np.array([[0,-1,0], 
                              [-1, 5,-1],
                              [0,-1, 0]])
    
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    return sharpened

def Blur(img):
    kernel_3x3 = np.ones((7, 7), np.float32) / 49
    # We apply the filter and display the image
    blurred = cv2.filter2D(img, -1, kernel_3x3)
    return blurred
    
# display
img = cv2.imread("3.jpg")

fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(10,10))
ax[0][0].imshow(img, vmin=0, vmax=255)#row=0, col=0
ax[0][0].set_title("Original Image")


ax[0][1].imshow(Blur(img), vmin=0, vmax=255)#row=0, col=1
ax[0][1].set_title("Blured Image")

ax[1][0].imshow(Sharpen(img), vmin=0, vmax=255)#row=0, col=0
ax[1][0].set_title("Sharpen Image")


ax[1][1].imshow(Cartoonize(img), vmin=0, vmax=255)#row=0, col=1
ax[1][1].set_title("Cartoonized Image")

ax[2][0].imshow(EdgeDetection(img,15,3), vmin=0, vmax=255)#row=0, col=0
ax[2][0].set_title("Edges")

plt.show()


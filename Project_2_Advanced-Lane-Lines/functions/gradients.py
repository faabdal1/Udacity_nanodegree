import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
        
    #Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        #Calculate the derivative in the x or y direction (the 1, 0 at the end denotes x direction):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
        
    #Calculate the absolute value of the x derivative:
    abs_sobel = np.absolute(sobel)
        
    #Convert the absolute value image to 8-bit:
    #Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        
        
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return sbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
        
    #covert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
        
    #3) calculate absolute of both sobels
    abs_sobel = np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
        
    #Convert the absolute value image to 8-bit:
    #Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
        
        #covert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
        
        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_grad = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        # 5) Create a binary mask where direction thresholds are met
    binary = np.zeros_like(dir_grad)
    binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
        
    return binary

def s_select(img, thresh=(100, 255)):
        # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
        # 3) Return a binary image of threshold result
  
    return (S > thresh[0]) & (S <= thresh[1])

        
def l_select(img, thresh=(120, 255)):
            # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
    L = hls[:,:,1]
        # 3) Return a binary image of threshold result
        
    return (L > thresh[0]) & (L <= thresh[1])


def gray_tresh(img, thresh=(0, 255)):
        # 1) Convert to gray space
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
        # 2) Create empty image of same size 
    binary = np.zeros_like(gray)
        # 3) Return a binary image of threshold result
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    return binary

def hls_l_xgrad(img, sobel_kernel=3, thresh=(0, 255)):
        # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
    l_chan = hls[:,:,1]
    sobel = cv2.Sobel(l_chan, cv2.CV_64F, 1, 0, sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # 3) Return a binary image of threshold result
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        
    return binary

def my_thr(img):
        
    # apply gradient threshold on the horizontal gradient
    gradx = abs_sobel_thresh(img, 'x', thresh=(10, 100))
    
    # apply gradient direction threshold so that only edges closer to vertical are detected.
    dir_binary = dir_threshold(img, thresh=(np.pi/6, np.pi/2))
    
    # combine the gradient and direction thresholds.
    combined_condition = ((gradx == 1) & (dir_binary == 1))
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    color_combined = np.zeros_like(gray)

    
    # S channel performs well for detecting bright yellow and white lanes
    s_condition = s_select(img, thresh=(100, 255))
    
    # L channel to avoid pixels which have shadows and as a result darker.
    l_condition = l_select(img, thresh=(120, 255))


    # combine all the thresholds
    # And it should also have a gradient, as per our thresholds
    color_combined[(l_condition) & (s_condition | combined_condition)] = 1
    
    return color_combined


##### Enable debugging and testing below ######
debugging = False

#Debugging Part
if debugging:

    # Read in a test image
    image = cv2.cvtColor(cv2.imread('../test_images/test4.jpg'), cv2.COLOR_RGB2BGR)


    # Choose a Sobel kernel size
    ksize = 5 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gray = gray_tresh(image, thresh=(150, 240))
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.2, 1))
    s_binary = s_select(image, thresh=(90, 255))
    l_xgrad = hls_l_xgrad(image, sobel_kernel=5, thresh=(20, 100))




    combined1 = np.zeros_like(dir_binary)
    combined1=s_binary | l_xgrad == 1
    combined2 = np.zeros_like(dir_binary)
    combined2[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1


    #combined1[((grady == 1) | (gradx == 1) | (hls_binary == 1) | (gray == 1))] = 1
    #combined1[(gradx == 1) | (grady == 1) | (mag_binary == 1) | (hls_binary == 1)] = 1
    #combined2 = np.zeros_like(dir_binary)
    #combined2[(gradx == 1) & (grady == 1) | (mag_binary == 1) | (hls_binary == 1)] = 1
    #combined3 = np.zeros_like(dir_binary)
    #combined3[(gradx == 1) | (grady == 1) & (mag_binary == 1) | (hls_binary == 1)] = 1
    #combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1


    f, axarr = plt.subplots(3,2)
    axarr[0,0].imshow(l_xgrad)
    axarr[0,1].imshow(grady)
    axarr[1,0].imshow(combined1)
    axarr[1,1].imshow(combined2)
    axarr[2,0].imshow(hls_binary)
    axarr[2,1].imshow(dir_binary)

    
    plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Read in an image
image = mpimg.imread('test_images/test5.jpg')


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

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    # 3) Return a binary image of threshold result
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    return binary

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(30, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

hls_binary = hls_select(image, thresh=(0, 115))


#To test combinations
combined0 = np.zeros_like(dir_binary)
combined0[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

combined1 = np.zeros_like(dir_binary)
combined1[((gradx == 1))] = 1

combined2 = np.zeros_like(dir_binary)
combined2[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) & (hls_binary == 1)] = 1

combined3 = np.zeros_like(dir_binary)
combined3[((gradx == 1) & (grady == 0)) | ((mag_binary == 1) & (dir_binary == 1)) & (hls_binary == 1)] = 1


#fig = plt.figure()
#plt.imshow(combined, cmap='gray')
#fig = plt.figure()
#plt.imshow(combined2, cmap='gray')


f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(combined0)
axarr[0,1].imshow(combined1)
axarr[1,0].imshow(combined2)
axarr[1,1].imshow(combined3)


#f.tight_layout()
#ax1[0,0].imshow(combined, map='gray')
#ax1.set_title('Original Image', fontsize=50)
#ax1[0,1].imshow(combined2, cmap='gray')
#ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#plt.imshow(combined, cmap='gray')
#plt.imshow(hls_binary, cmap='gray')
plt.show()

print("Done")
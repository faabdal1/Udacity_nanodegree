import numpy as np
import cv2, glob, time, pickle, os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pylab as pl
import pickle

from functions.calibration import camera_calibration
from functions.gradients import Gradients
from functions.lane_hist import hist, fit_polynomial

# To run camera calibration and generate calibration_parameters.p
#camera_calibration(visualization = False)

# If the the calibration_parameters.p exist, reads the parameters and continues
if os.path.isfile('calibration_parameters.p'):
    mtx = pickle.load(open("calibration_parameters.p", "rb"))["mtx"]
    dist = pickle.load(open("calibration_parameters.p", "rb"))["dist"]
    print("Calibration parameters loaded")
else:
    print ("Calibration_parameters.p files does not exist, running calibration to generage the file")
    camera_calibration(visualization = False)
    mtx = pickle.load(open("calibration_parameters.p", "rb"))["mtx"]
    dist = pickle.load(open("calibration_parameters.p", "rb"))["dist"]
    print("Calibration parameters loaded")

# Print the distort coefficient parameters
print("Dist: " + str(dist))
print("Mtx: " + str(mtx))



#### Load test image or video to read frames ####

# load video files
#cam = cv2.VideoCapture("project_video.mp4")
#ret,frame = cam.read()

# load image files
image = cv2.cvtColor(cv2.imread('test_images/test7.png'), cv2.COLOR_RGB2BGR)
image = cv2.undistort(image, mtx, dist, None, mtx)


# Apply selected gradients to the image 
hls_s = Gradients.hls_select(image, thresh=(170, 255))
hls_l_gradx = Gradients.hls_l_xgrad(image, 5, thresh=(20, 255))

combined = np.zeros_like(image)
combined = hls_s | hls_l_gradx 

#Define region of interest and apply to edges
imshape = image.shape
print(imshape)
a = (int(imshape[1]*0.19), int(imshape[0]-40))
b = (int(imshape[1]*0.462), int(imshape[0]*0.63))
c = (int(imshape[1]*0.538), int(imshape[0]*0.63))
d = (int(imshape[1]*0.83), int(imshape[0]-40))

vertices = np.array([[a, b, c, d]], dtype=np.int32)
print("Vertices: " + str(vertices))

# Draw region for perspective tranformation 
cervena = image.copy()
cervena = cv2.line(cervena, a, b, color=[255, 0, 0], thickness=3)
cervena = cv2.line(cervena, b, c, color=[255, 0, 0], thickness=3)
cervena = cv2.line(cervena, c, d, color=[255, 0, 0], thickness=3)
cervena = cv2.line(cervena, d, a, color=[255, 0, 0], thickness=3)
f, axarr = plt.subplots(3,2)
axarr[1,0].imshow(cervena)


src = np.float32([[a],[b],[c],[d]])
img_size = (image.shape[1], image.shape[0])
print(img_size)

dst = np.float32(
	[[(imshape[1] / 4), imshape[0]],
    [(imshape[1] / 4), 0],
    [(imshape[1] * 3 / 4), 0],
	[(imshape[1] * 3 / 4), imshape[0]]])



# Use cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# Use cv2.warpPerspective() to warp your image to a top-down view
warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)

axarr[1,1].imshow(warped)
plt.show()


# Create histogram of image binary activations
histogram = hist(warped)

# Visualize the resulting histogram
plt.plot(histogram)
plt.show()

out_img, right_fitx, left_fitx, ploty = fit_polynomial(warped)

plt.imshow(out_img)
plt.show()

# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(M), (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
plt.figure()
plt.imshow(result)
plt.show()


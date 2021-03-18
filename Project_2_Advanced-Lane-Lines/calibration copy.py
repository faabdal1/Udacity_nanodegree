import numpy as np
import cv2, glob, time, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Number of inside corners in calibration chessboard
nx = 9 #number of inside corners in x
ny = 6 #number of inside corners in y

# To read an images 
images = glob.glob('camera_cal/calibration*.jpg')
#img = cv2.imread(fname)

# Lists to store object points and image points from all images
objpoints = []
imgpoints = []

# Preparation of object points 9x6 size
objp = np.zeros((ny*nx,3), np.float32)
#print(objp)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x ,y coordinates
#print(objp)

# Iterate through list of images
for idx, fname in enumerate(images):
    img = cv2.imread(fname)    

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners; returns boolen and list with corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners and show the image
    if ret == True:
        #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        #plt.imshow(img)
        #plt.show()

        # Append found corners to the imgpoints list, objpoints are always same since they represent same chessboard
        imgpoints.append(corners)
        objpoints.append(objp)


img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

#print(ret)
#print(mtx)
#print(dist)
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('test_undist.jpg',dst)
    
#undist = cv2.undistort(img, mtx, dist, None, mtx)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open( "calibration_parameters.p", "wb" ) )

print(pickle.load(open("calibration_parameters.p", "rb")))
    
def calibration (self, path, nx, ny):
    

    return True
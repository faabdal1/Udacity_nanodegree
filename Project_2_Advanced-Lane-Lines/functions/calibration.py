import numpy as np
import cv2, glob, time, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Camera calibration function, no input parameters are required if the images and chessboard follows default setting
#INPUTS:    path ... folder with calibration images, default: camera_cal/calibration*.jpg
#           nx ... number of inside corners in x, default: 9
#           ny ... number of inside corners in y default: 6
#           visualization ... show output chessboard plot over the images, default = False
#           test_image ... destination folder and name of the test image to calculate distortion coefficients default = 'camera_cal/test_image.jpg'
#OUTPUT:    Saves calibration_parameters.p file with distortion coefficients and returns True

def camera_calibration (path = 'camera_cal/calibration*.jpg', nx = 9, ny = 6, visualization = False, test_image = 'camera_cal/test_image.jpg'):

    # Number of inside corners in calibration chessboard
    #nx = 9 #number of inside corners in x
    #ny = 6 #number of inside corners in y

    # To read an images 
    images = glob.glob(path)
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
            #if we want to have visualization of the chessboard over the images (default false)
            if visualization:
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.ion()
                plt.imshow(img)
                plt.show()
                plt.pause(0.5)

            # Append found corners to the imgpoints list, objpoints are always same since they represent same chessboard
            imgpoints.append(corners)
            objpoints.append(objp)

    # Read test image 
    img = cv2.imread(test_image)
    # Get size of the test image
    img_size = (img.shape[1], img.shape[0])

    # Applie calibrateCamera function to calculate distortion coeficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Logging prints
    #print("Ret: " + str(ret))
    #print("Mtx: " + str(mtx))
    #print("Dist: " + str(dist))

    # Apply undistort function to apply calculated coefficients and undistor the image
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Save the undistort image to a root
    cv2.imwrite('test_undist.jpg',dst)
        
    # Save the camera calibration result for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open( "calibration_parameters.p", "wb" ) )

    # Option to check what is saved in the file
    #print(pickle.load(open("calibration_parameters.p", "rb")))

    # Message with successful calibration process
    print("Calibartion successful, calibration coefficients are stored in calibration_parameters.p file!")
    return True 
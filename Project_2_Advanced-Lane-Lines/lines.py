import numpy as np
import cv2, glob, time, pickle, os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pylab as pl
import pickle
from moviepy.editor import VideoFileClip

from functions.gradients import my_thr
from functions.calibration import camera_calibration


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        
        #leftx from last frame
        self.last_fit_left = [np.array([False])]  
        #rightx from last frame
        self.last_fit_right = [np.array([False])]  
 
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  

        # Initiates calibration process or loads calibration parameters
        # If the the calibration_parameters.p exist, reads the parameters and continues
        if os.path.isfile('calibration_parameters.p'):
            self.mtx = pickle.load(open("calibration_parameters.p", "rb"))["mtx"]
            self.dist = pickle.load(open("calibration_parameters.p", "rb"))["dist"]
            print("Calibration parameters loaded")
        else:
            print ("Calibration_parameters.p files does not exist, running calibration to generage the file")
            camera_calibration(visualization = False)
            self.mtx = pickle.load(open("calibration_parameters.p", "rb"))["mtx"]
            self.dist = pickle.load(open("calibration_parameters.p", "rb"))["dist"]
            print("Calibration parameters loaded")

        # Print the distort coefficient parameters
        #print("Dist: " + str(dist))
        #print("Mtx: " + str(mtx))
        
    # function to process image and return warped
    def img_process(self, image):
        # Undistort image based on calibration parameters 
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        # to apply gradients to the image and get binary image
        combined = my_thr(image)

        #plt.figure()
        #plt.imshow(combined)
        #plt.show()

        #size of current frame
        imshape = image.shape
        #print(imshape)

        #Define region which is rectangle from top view
        a = (int(imshape[1]*0.19), int(imshape[0]-40))
        b = (int(imshape[1]*0.462), int(imshape[0]*0.63))
        c = (int(imshape[1]*0.538), int(imshape[0]*0.63))
        d = (int(imshape[1]*0.83), int(imshape[0]-40))

        vertices = np.array([[a, b, c, d]], dtype=np.int32)
        #print("Vertices: " + str(vertices))

        # We can draw region for perspective tranformation 
        cervena = image.copy()
        cervena = cv2.line(cervena, a, b, color=[255, 0, 0], thickness=3)
        cervena = cv2.line(cervena, b, c, color=[255, 0, 0], thickness=3)
        cervena = cv2.line(cervena, c, d, color=[255, 0, 0], thickness=3)
        cervena = cv2.line(cervena, d, a, color=[255, 0, 0], thickness=3)
        
        


        #define source region for transforamtion
        src = np.float32([a,b,c,d])

        #define destination region for transforamtion
        dst = np.float32(
            [[(imshape[1] / 4), imshape[0]],
            [(imshape[1] / 4), 0],
            [(imshape[1] * 3 / 4), 0],
            [(imshape[1] * 3 / 4), imshape[0]]])
        dst1 = np.float32(
            [((imshape[1] / 4), imshape[0]),
            ((imshape[1] / 4), 0),
            ((imshape[1] * 3 / 4), 0),
            ((imshape[1] * 3 / 4), imshape[0])])

        # Use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(combined, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        return warped, M

    #find polynomial representation of the two lines
    def fit_polynomial(self, binary_warped):
        if self.last_fit_left[0] == False or self.last_fit_right[0] == False :
            
            # Find our lane pixels first
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

            #Fit a second order polynomial to each using `np.polyfit` ###
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            self.last_fit_left = left_fit
            self.last_fit_right = right_fit

            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            try:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            except TypeError:
                # Avoids an error if `left` and `right_fit` are still none or incorrect
                print('The function failed to fit a line!')
                left_fitx = 1*ploty**2 + 1*ploty
                right_fitx = 1*ploty**2 + 1*ploty

            ## Visualization ##
            # Colors in the left and right lane regions
            #out_img[lefty, leftx] = [255, 0, 0]
            #out_img[righty, rightx] = [0, 0, 255]

            # Plots the left and right polynomials on the lane lines
            #plt.plot(left_fitx, ploty, color='yellow')
            #plt.plot(right_fitx, ploty, color='yellow')
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(out_img, "Left: f(y) = {}y**2 + {}y + {}".format(round(left_fit[0],2),round(left_fit[1],2),round(left_fit[2],2)), (150,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #cv2.putText(out_img, "Right: f(y) = {}y**2 + {}y + {}".format(round(right_fit[0],2),round(right_fit[1],2),round(right_fit[2],2)), (400,500), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #plt.imshow(out_img)
            #plt.show()

        else:
            # HYPERPARAMETER Choose the width of the margin around the previous polynomial to search
            margin = 100

            left_fit = self.last_fit_left
            right_fit = self.last_fit_right
            # Grab activated pixels
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            #Set the area of search based on activated x-values
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                            left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                            left_fit[1]*nonzeroy + left_fit[2] + margin)))
            right_lane_inds =  ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                            right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                            right_fit[1]*nonzeroy + right_fit[2] + margin)))
            
            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            self.last_fit_left = left_fit
            self.last_fit_right = right_fit

            img_shape = binary_warped.shape
            # Generate x and y values for plotting
            ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
            # Calc both polynomials using ploty, left_fit and right_fit
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            #window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
            #                        ploty])))])
            #left_line_pts = np.hstack((left_line_window1, left_line_window2))
            #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
            #                        ploty])))])
            #right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            #color_warp = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)     
            # Plot the polynomial lines onto the image
            #plt.plot(left_fitx, ploty, color='yellow')
            #plt.plot(right_fitx, ploty, color='yellow')
            ## End visualization steps ##       
        return right_fitx, left_fitx, ploty


    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 12
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 30

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            #Find the four below boundaries of the window 
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window 
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window 
            ### (`right` or `leftx_current`) on their mean position 
            if len(good_left_inds) > (minpix):
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > (minpix):
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def draw_carpet(self, image, warped, right_fitx, left_fitx, ploty, M):

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(M), (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result

    def calc_curv(self, right_fitx, left_fitx, ploty):   
        y_eval = np.max(ploty)
        ym_per_pix = 30/700 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # width of lane per pixel (in y dimension)

        # Implement the calculation of the left and right lines 
        left_curverad = ((1 + (2*self.last_fit_left[0]*y_eval*ym_per_pix + self.last_fit_left[1])**2)**1.5) / (np.absolute(2*self.last_fit_left[0]))  
        right_curverad = ((1 + (2*self.last_fit_right[0]*y_eval*ym_per_pix + self.last_fit_right[1])**2)**1.5) / np.absolute(2*self.last_fit_right[0])
        
        #print("left: " + str(left_curverad))
        #print("right: " + str(right_curverad)
        self.radius_of_curvature = round((left_curverad + right_curverad)/2)

        mean = ((right_fitx[0] + left_fitx[0])/2)-640
        #print(mean)
        offset = xm_per_pix * (mean/2)
        #print(offset)

        return round(offset,2)


    def sanity_check(self,right_fitx, left_fitx):

        # To check width of the line
        # Take x value of left and right line
        botom = right_fitx[10] - left_fitx[10]
        #print(right_fitx[10])
        #print(left_fitx[10])

        #print(botom)
        #print(top)

        # Check condition
        if ((botom > 600) and (botom <= 800)):
            self.detected = True
            return True
        else:
            self.detected = False
            return False

    def pipeline(self, image):
        # To process raw frame and receive warped binary and M transformation matrix
        warped, M = self.img_process(image)
        # To identify lanes and fit polynomial to them.
        right_fitx, left_fitx, ploty = self.fit_polynomial(warped)
        #plt.figure()
        #plt.imshow(color_warp)
        #plt.show()

        # To calculate lane curvature and position of the vehicle 
        offset = self.calc_curv(right_fitx, left_fitx, ploty)

        # Sanity check, if the lanes are not parallel, then search is sent back to sliding window
        if self.sanity_check(right_fitx, left_fitx) == False:
            print("search reset")
            self.last_fit_left = [np.array([False])]
            self.last_fit_left = [np.array([False])]
            right_fitx, left_fitx, ploty = self.fit_polynomial(warped)
            self.calc_curv(right_fitx, left_fitx, ploty)
            self.sanity_check(right_fitx, left_fitx)
        #defined font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # include information about curvature and offset into frames
        if self.detected == True:
            result = self.draw_carpet(image, warped, right_fitx, left_fitx, ploty, M)
            cv2.putText(result, "Curvature: " + str(self.radius_of_curvature), (500,50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(result, "Vehicle is {}m left of center".format(str(offset)), (500,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            result = self.draw_carpet(image, warped, right_fitx, left_fitx, ploty, M)
            cv2.putText(result, "Bad detection ", (500,50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print("no lines")
        return result

if __name__ == '__main__':
    
    ### What to process: im=images , vid=videos
    process = "im"
    #process = "vid"

    if process == "im":

        # Individual images:
        list = os.listdir("test_images/")
        #list = ["straight_lines2.jpg"]
        for im_name in list:
            l = Line()
            if im_name ==".DS_Store": #filter unwanted mac files
                continue
            else:
                img = l.pipeline(cv2.cvtColor(cv2.imread('test_images/{}'.format(im_name)), cv2.COLOR_RGB2BGR))
                cv2.imwrite(r"output_images/{}".format(im_name), (cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) )
    elif process == "vid":
        
        l = Line()
        white_output = 'project_video_output.mp4'
        clip = VideoFileClip("project_video.mp4")
        #clip = VideoFileClip("harder_challenge_video.mp4").subclip(5,10)
        white_clip = clip.fl_image(l.pipeline) 
        white_clip.write_videofile(white_output, audio=False)
    else:
        print("Choose data source")

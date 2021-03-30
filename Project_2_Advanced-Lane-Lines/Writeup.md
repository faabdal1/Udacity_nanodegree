**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_im/im1.png "Undistorted"
[image2]: ./writeup_im/im2.png "Distortion correction"
[image3]: ./writeup_im/im3.png "Binary Example"
[image4]: ./writeup_im/im4.png "Warp Example"
[image5]: ./writeup_im/im5.png "Polynomial"
[image6]: ./writeup_im/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code in "function/calibration.py" deals with the calibration process.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The chessboard used for this project was 9x6.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Calibration][image1]

The calibration scripts saves calibration parameters `mtx` and `dist` into `calibration_parameters.p` file and  it is used in pipeline to correct distortion.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The parameters obtained in previous calibration step are loaded and used as input for correction every image or frame. Function `cv2.undistort()` is used to correct distortion of every image or video frame. Inputs parameters are image to correct, calibration matrix and distortion. Return is then corrected image.

To demonstrate this step, there is example of distortion correction here:
![Distortion correction][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding function at lines 127 through 154 in `functions/gradients.py`).

I combine two gradient thresholds. Sobel gradient in X direction with thresholds min: 10, max: 100. Second directional threshold with min: pi/6, max: pi/2. These two gradients are combined together.

Secondly, the image was transformed into HSL color space. The S and L channels were used to picked. S channel because it picks well on yellow lanes and L channel because it filters well shadows and darker pixels.

All thresholds are then combined into empty binary image and returned to the pipeline. 

Here's an example of my output for this step.

![Binary example][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `img_process()`, which appears in lines 60 (transformation starts on line 73) through 115 in the file `lines.py`.  The `img_process()` function takes as inputs a binary image (`combined`), which has same size as original image. I chose the hardcode the source (`src`) and destination (`dst`) points in the following manner:

```python
imshape = image.shape

a = (int(imshape[1]*0.19), int(imshape[0]-40))
b = (int(imshape[1]*0.462), int(imshape[0]*0.63))
c = (int(imshape[1]*0.538), int(imshape[0]*0.63))
d = (int(imshape[1]*0.83), int(imshape[0]-40))

src = np.float32([[a],[b],[c],[d]])

dst = np.float32(
    [[(imshape[1] / 4), imshape[0]],
    [(imshape[1] / 4), 0],
    [(imshape[1] * 3 / 4), 0],
    [(imshape[1] * 3 / 4), imshape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 243, 680      | 320, 720      | 
| 591, 453      | 320, 0        |
| 688, 453      | 960, 0        |
| 1062, 680     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Note: The perspective transformation in the actual script is performed on a binary input, but for purpose of demonstration, colorful image was used.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The function `find_lane_pixels` to identify lines is located in `lines.py` on from lines 223 to 304. This function identify lines without previous identification. This function is part of the `fit_polynomial` which takes output from `find_lane_pixels` and calculates polynomial. It also checks whether the lines were identified in previous frame and if yes, then it skips to second half od the fit polynomial (lines 158 - 223 in `lines.py`) and searches for lines based on the polynomial from previous frame.

A) No detected lines in previous frame (applicable for single images)

- if no lines were detected in previous frame or when it is image the function `find_lane_pixels` goes on. This function creates histogram and divides it into two half (left and right). Peak on right side and peak on left side defines middle point for sliding window.
- There are defined Hyperparemeters: number of sliding window (12), width of sliding window (100px) and minimal number of pixel to detect line (30px).
- Then define lists to store detected lines for left and right side and setting height of window according to the number of windows.
- then according to number of windows (rows), functions searches for lines according to the defined hyperparameters and append the list of pixel to the list and recenter the window to the position of last detected line 
- optional: we can draw the window over the binary image.
- concatenate the line to transform the array of pixel to indices
- last step is to extract left and right line pixel positions and return results
- to fit polynomial, function `np.polyfit()` is used and it calculates second order polynomial based on the detected lines in previous step.

B) Detected lines in previous frame (lines 158 - 223 in `lines.py`)
- The lines are detected base on the polynomial from previous step.
- the line of code are on line 159 - 182 in `lines.py`
- it searches for lines in are of the polynomial from previous frame with predefined margin (100px)
- it again extract pixel position of left and right lanes and calculates polynomial


![Polynomial][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

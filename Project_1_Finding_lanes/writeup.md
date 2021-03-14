# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[grayscale]: ./writeup/grayscale.png "Grayscale"
[gausian]: ./writeup/blur_gray.png "Gausian smoothing"
[canny]: ./writeup/canny.png "Canny edge detection"
[mask]: ./writeup/mask.png "Polygon of interest"
[canny_mask]: ./writeup/canny_mask.png "Canny edge detection and filtered region of interest"
[raw_lines]: ./writeup/raw_lines.png "Raw lines drawn on a blank image"
[raw_lines_image]: ./writeup/raw_lines_image.png "Raw lines image drawn over original image"
[lines]: ./writeup/lines.png "Averaged and interpolated lines drawn on a blank image"
[lines_image]: ./writeup/lines_image.png "Lines image drawn over original image"

---

### Reflection

### 1. Pipeline

The pipeline consists of several steps. 

1. step is to load the picture (sigle picture or in loop from video) and convert it into grayscale

![Grayscale pic][grayscale]

2. step is to apply Gausian smoothing where I chose kernel size 5 to reduce noise and enhance image structures.

![Gausian smoothing][gausian]

3. step is to define parameters for canny edge detection and apply to the image 

![Canny detection][canny]

4. step is to define polygon of interest and apply it to the edges detected by Canny detection. This limits field of view to part in front of a vehicle and reduces noice from surounding. The polygon of interest is defined based on the resolution of the image, to make sure that it stays proportional to the image and camtures relevant area.

![Mask][mask]
![Canny_mask][canny_mask]


5. step is to apply Hough line detection to filter relevat segments which represent borders of lanes. The individual parameters of Hough line detection efect the sensitivity and final lane detection. For my lane detecition, I have used rho = 1, treshold = 20, min_line_length = 25, max_line_gap = 50. Function returns array of lines endpoints.

6. a) step interate through the lines and draw them on a blank image. The initial function draws a raw lines without aditional processing as it is show on picture below. The image is then drawn on the original image (semi-transparent) and saved or added to the new video stream.

![Raw_lines][raw_lines]
![Raw_lines_image][raw_lines_image]

6. b) in order to improve detection, draw function was improved. The lines detected by Hough line fuction were separated according to their slope to decide which belong on left and right side. The mean position and slope of right lines and also mean position and slope of left lines were found. These two lines were then interpolated and the linear lines were drawn on a blank picture. The picture is then drawn (semi-transparant) on the original image and saved or added to the new video stream.

![Lines][lines]
![Lines_image][lines_image]

7. testing of the line detection function on various video stream and check outcome.

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is detection in different light condition or weather conditions. Espetially night time brings new challenges. 

Selection of parameters for Canny function and Hough line detection was done based on personal feeling. It might not cover the best possible results.

Another shortcoming is that the function expects that the lane marking is straight. The it applies only linear interpolation and does not cosider turns.

The function does not uses information from previous frame or it does not stabilises the detected lines.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to include infromation from previous frames and to stabilises line detection. Currently some frames causes suddent jumps in line detection.

Another improvment should be to detect bend lines in curves and to interpolate using polynomial function.

Third possible improvement could be to add confident level to describe how good the line detection is and to estimate lane marking in cases that one frame does not get any detection.

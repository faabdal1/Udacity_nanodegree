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

1. step is to load the picture (single picture or in a loop from video) and convert it into grayscale

![Grayscale pic][grayscale]

2. step is to apply Gaussian smoothing where I chose kernel size 5 to reduce noise and enhance image structures.

![Gausian smoothing][gausian]

3. step is to define parameters for canny edge detection and apply them to the image 

![Canny detection][canny]

4. step is to define a polygon of interest and apply it to the edges detected by Canny detection. This limits the field of view to part in front of a vehicle and reduces noise from surrounding. The polygon of interest is defined based on the resolution of the image, to make sure that it stays proportional to the image and captures the relevant area.

![Mask][mask]
![Canny_mask][canny_mask]


5. step is to apply Hough line detection to filter relevant segments which represent borders of lanes. The individual parameters of Hough line detection effect the sensitivity and final lane detection. For my lane detection, I have used rho = 1, threshold = 20, min_line_length = 25, max_line_gap = 50. The function returns an array of lines endpoints.

6. a) step iterates through the lines and draw them on a blank image. The initial function draws raw lines without additional processing as is shown in the picture below. The image is then drawn on the original image (semi-transparent) and saved or added to the new video stream.

![Raw_lines][raw_lines]
![Raw_lines_image][raw_lines_image]

6. b) to improve detection, the draw function was improved. The lines detected by Hough line function were separated according to their slope to decide which belong on the left and right side. The mean position and slope of right lines and also mean position and slope of left lines were found. These two lines were then interpolated and the linear lines were drawn on a blank picture. The picture is then drawn (semi-transparent) on the original image and saved or added to the new video stream.

![Lines][lines]
![Lines_image][lines_image]

7. testing of the line detection function on various video stream and check outcome.

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is detection in different light condition or weather conditions. Especially night time brings new challenges. 

Selection of parameters for Canny function and Hough line detection was done based on personal feeling. It might not cover the best possible results.

Another shortcoming is that the function expects that the lane marking is straight. It applies only linear interpolation and does not consider turns.

The function does not use information from a previous frame or it does not stabilise the detected lines.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to include information from previous frames and to stabilises line detection. Currently, some frames cause sudden jumps in lines detection.

Another improvement should be to detect bend lines in curves and to interpolate using a polynomial function.

The third possible improvement could be to add a confident level to describe how good the line detection is and to estimate lane marking in cases that one frame does not get any detection.
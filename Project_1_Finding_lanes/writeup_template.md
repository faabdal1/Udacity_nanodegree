# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

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
[canny_mask]: ./writeup/mask.png "Canny edge detection and filtered region of interest"


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


5. step is to apply Hough line detection to filter relevat segments which represent borders of lanes. The individual parameters of Hough line detection efect the sensitivity and final lane detection. For my lane detecition, I have used rho = 1, treshold = 20, min_line_length = 25, max_line_gap = 50.




In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

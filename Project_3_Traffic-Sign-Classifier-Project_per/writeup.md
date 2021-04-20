# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write_up_img/1.jpeg "Random image"
[image2]: ./write_up_img/2.jpeg "Distribution"
[image3]: ./write_up_img/3.jpeg "Normalization"
[image4]: ./test_img/01.jpg "Traffic Sign 1"
[image5]: ./test_img/02.jpg "Traffic Sign 2"
[image6]: ./test_img/03.jpg "Traffic Sign 3"
[image7]: ./test_img/04.jpg "Traffic Sign 4"
[image8]: ./test_img/05.jpg "Traffic Sign 5"
[image9]: ./write_up_img/6.jpeg "First layer"
[image10]: ./write_up_img/7.jpeg "Second layer"

---
### Writeup / README

Here is a link to my [project code](https://github.com/faabdal1/Udacity_nanodegree/blob/master/Project_3_Traffic-Sign-Classifier-Project_per/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
    * 34799 images    
    * `len(X_train)` 
* The size of the validation set is ?
    * 4410 images
    * `len(X_valid)`
* The size of test set is ?
    * 12630
    * `len(X_test)`
* The shape of a traffic sign image is ?
    * (32,32)
    * `(X_train.shape[1],X_train.shape[2])`
* The number of unique classes/labels in the data set is ?
    * 43
    * `max(y_valid)+1`


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

* I randomly select one image from the training set and visualize it.

![Random image from the training set][image1]

* In the bach chart below, I plot distribution of classes for training, validation and test sets. It is obvious that certain classes (e.g., class 2) have many more training images in the sets then e.g., class 0.

![Image classes distributions for training, validation and test sets][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the data sets using: `(img-128)/128`

![Normalization][image3]

I normalized the image data to make sure that I use data of similar magnitude and not to add very small number to large and vice versa.

I have tried to convert the data sets to grayscale before normalizing them but it let to lower accuracy for validation and test data set. That is why I decided to keep color data.

The difference between the original data set and the augmented image data has the mean zero and equal variance.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU          		|           									|
| Dropout				| I set 0.6     								|
| Max pooling			| 2x2 stride,  outputs 5x5x16     				|
| Flatten				| Flatten into vector, output: 400 				|
| Fully connected 1D	| Output: 120    					    		|
| RELU          		|           									|
| Dropout				| I set 0.6     								|
| Fully connected 1D	| Output: 84  							    	|
| RELU          		|           									|
| Dropout				| I set 0.6     								|
| Fully connected		| Final output: 43 ==> same as number or classes|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used started with 10 epochs and then increased to final 15 which were enough to reach sufficient accuracy. 
I used default batch size of 128 and learning rate of 0.001. I also used keep coef for dropout of 0.6

I have train the model observed training accuracy and validation accuracy in each iteration. Once the accuracy was reached on validation set, I saved the model and I tested the model on test set.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.954
* test set accuracy of 0.934

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * The first architecture I tried was pure Lenet architecture without any modification and basic parameters from the course 
* What were some problems with the initial architecture?
    * The main problem was final accuracy even below 0.9 
    * I have also started with wrong output of logits, I kept 10 insted of 43 which let to very poor performance and did not really work 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * The main adjustment of the model architecture was to add dropout. That improved accuracy dramatically
    * I kept the images in RGB instead of grayscale
    * The difference between train, validation and test accuracy was acceptable, so there wasn't need to adjust anything else
* Which parameters were tuned? How were they adjusted and why?
    * I tuned parameter for dropout to 0.6
    * Increased epochs to 15
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Dropout helped the model to not overfit and increased accuracy on the validation set and also test set
    * Another choice was to keep the data sets in RGB instead of converting them into Grayscale. It increased the accuracy because I did not lose some information from RGB data images 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I did not choose the most difficult traffic signs but there were some issues.

The most problematic was second sign, 100 km/h speed limit. That image is not very clear and it let to very low accuracy but end the end right classification.

The first sign did not have high accuracy too. I expect that it was due to small sample of this sign in training set. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        |   Accuracy    |
|:---------------------:|:-------------------------:|:-------------:| 
| Children crossing     | Children crossing 		|      0.76     | 
| 100 km/h     			| 100 km/h 					|      0.20     |
| No entry  			| No entry					|      0.99     |
| Stop sign	      		| Stop sign				    |      1.0      |
| Keep right			| Keep right			    |      1.0      |


The model was able to correctly guess all 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.4%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is not very sure that this is a children crossing sign (probability of 0.76). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .76         			| Children crossing   							| 
| .12     				| Right-of-way at the next intersection			|
| .07					| Bicycles crossing								|
| .01	      			| Vehicles over 3.5 metric tons prohibited		|
| .01				    | Road narrows on the right						|


For the second image, the model is not sure that this is a 100 km/h sign (probability of 0.20). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .208        			| Speed limit (100km/h)  						| 
| .205     				| Speed limit (30km/h)	                    	|
| .14					| Roundabout mandatory							|
| .11	      			| Speed limit (80km/h)	                    	|
| .10				    | Speed limit (120km/h)	        				| 

For the third image, the model is sure that this is a No entry sign (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| No entry                						| 
| .00     				| Speed limit (20km/h)	                    	|
| .00					| Speed limit (30km/h)							|
| .00	      			| Stop                                      	|
| .00				    | Yield                         				| 
 
For the forth image, the model is sure that this is a Stop sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Stop sign	              						| 
| .00     				| Speed limit (20km/h)	                    	|
| .00					| Speed limit (80km/h)							|
| .00	      			| No entry                                    	|
| .00				    | Speed limit (60km/h)	                        | 

For the last image, the model is sure that this is a Keep right sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Keep right              						| 
| .00     				| Turn left ahead   	                    	|
| .00					| Go straight or right							|
| .00	      			| Yield                                        	|
| .00				    | End of all speed and passing limits           | 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


On the image below is visualize output from 1st layer. It is still possible to recognize shape of the traffic sign. The network highted edges of the image. The output of the first layer was (5, 14, 14, 6)
![Visualization of the first layer][image9]

On the image below is visualize output from 2nd layer. The shape of the sign is complete gone and replaced by pixels. The output of the first layer was (5, 5, 5, 16)

![Visualization of the second layer][image10]



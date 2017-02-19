#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here is a link to my [project code](https://github.com/hrektts/CarND-Traffic-Sign-Classifier-Project/blob/submission/Traffic_Sign_Classifier.ipynb).

[//]: # (Image References)

[distribution]: ./fig/training_data_distribution.png "Distribution of training data"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[limit_30]: ./examples/limit_30_l.jpg "Limit 30 Sign"
[curve_left]: ./examples/curve_left_l.jpg "Curve Left Sign"
[wild_animal]: ./examples/wild_animals_crossing_l.jpg "Wild Animal Sign"
[stop]: ./examples/stop_l.jpg "Stop Sign"
[elderly_people]: ./examples/elderly_people_l.jpg "Elderly People Sign"

###Data Set Summary & Exploration

The code for this step is contained in the second to fourth code cell of the
IPython notebook.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The graph below shows what classes/labels are contained in the training data.
It is clear that the classes is not evenly distributed.

![Distribution of training data][distribution]

###Design and Test a Model Architecture

The code for preprocessing the data is contained in the fifth code cell of the
IPython notebook.

As a first step, I decided to subtract the mean of all pixels of the training set
to eliminate the effect of brightness difference among pictures.

Next, I normalized the value of the pixel between -1 and 1. This operation is not necessary
for the data set because the pixel of the images always have a value between 0 and 255.
However, I did it because the model created in the following steps can be used for other
data set.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer                 | Description                                   |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x3 RGB image                             |
| Convolution 5x5       | 1x1 stride, same padding, outputs 32x32x32    |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, same padding, outputs 32x32x64    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 16x16x64                 |
| Convolution 5x5       | 1x1 stride, same padding, outputs 16x16x128   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 8x8x128                  |
| Fully connected       | output 1024                                   |
| RELU                  |                                               |
| Dropout               | 25% (keep 75%)                                |
| Fully connected       | output 256                                    |
| RELU                  |                                               |
| Dropout               | 25% (keep 75%)                                |
| Fully connected       | output 34                                     |
| Softmax               |                                               |


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook.

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![Limit 30 Sign][limit_30] ![Curve Left Sign][curve_left]
![Wild Animal Sign][wild_animal] ![Stop Sign][stop]
![Elderly People Sign][elderly_people]

The first image might be difficult to classify because it is not facing right.
The second image might be difficult too because this class is trained with
a small amount of images compared to the other classes.
The third image is also considered to be difficult because it is a mirror image.
The fourth image might also be difficult because the outline of the sign is
different from ordinary.
The fifth image cannot be categolize correctly using the model trained above because
this kind of signs, which alert elderly people crossing, are not contained in
the training data. However, I selected it because I was interested in how it
to be classified.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image                 | Prediction                                  |
|:---------------------:|:---------------------------------------------:| 
| Limit 30 Sign         | Stop sign   									| 
| Curve Left Sign       | U-turn 										|
| Wild Animal Sign      | Yield											|
| Stop Sign             | Bumpy Road					 				|
| Elderly People Sign   | Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

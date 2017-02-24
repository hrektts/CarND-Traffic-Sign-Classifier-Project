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

As a first step, I decided to subtract the mean of all pixels of the training
set to eliminate the effect of brightness difference among pictures.

Next, I normalized the value of the pixel between -1 and 1. This operation is
not necessary for the data set because the pixel of the images always have a
value between 0 and 255.
However, I did it because the model created in the following steps can be used
for other data set.

To cross validate my model, I used provided validation data, which number of
images was 4410.

The code for my final model is located in the sixth cell of the ipython notebook.

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
| Fully connected       | output 43                                     |
| Softmax               |                                               |


The code for training the model is located in the seventh to eighth cell of the
ipython notebook.

To train the model, I used an adam optimizer. To select the batch size and
learning rate, I tested the value of 64, 128, 256 and 0.0001, 0.001, 0.01
respectively. As a result, I choosed 128 for the batch size and 0.001 for the
learning rate because the highest validation accuracy was achieved with the
values.

The code for calculating the accuracy of the model is located in the eighth to
ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.962
* test set accuracy of 0.950

To find the solution, I took an iterative approach.
First, I tested LeNet-5, which achieved about 85% of validation accuracy.
I thought the accuracy would be improved if the model could recognize
complicated figures because LeNet-5 was devised to identify characters.

Next my approach to improve the model was increasing depth of each layer. I did
this because I wanted my model to recognize more characteristics. As a result,
validation accuracy was improved to about 90%.

Finally, I increased the number of convolution layers because I wanted to
increase the granularity of the characteristics recognized by the model.
As mentioned in the lecture video “Visualizing CNNs”, each layer is trained to
learn different granularity of characteristics. I increased the layer because
I thought that the traffic signs is composed from multiple shapes with different
complexity. As a result, my model achieved 96% of validation accuracy.

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

The fifth image cannot be categorized correctly using the model trained above
because this kind of signs, which alert elderly people crossing, are not
contained in the training data. However, I selected it because I was interested
in how it to be classified.

The code for making predictions on my final model is located in the tenth cell of
the Ipython notebook.

Here are the results of the prediction:

| Image                        | Prediction                                |
|:----------------------------:|:-----------------------------------------:|
| Speed limit (30km/h)         | Speed limit (30km/h)                      |
| Dangerous curve to the left  | Dangerous curve to the left               |
| Wild animals crossing        | Children crossing                         |
| Stop                         | Stop                                      |
| Elderly people crossing      | Dangerous curve to the right              |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an
accuracy of 60%. This is worse than the accuracy of the test set. It is mainly
because I choosed the images which are not contained in the training set.

The mistakes for the third image might be improved by providing flipped image
when I augmented the training set. However, some classes of the images cannot
be flipped because the processed image will have different meanings.
For example, the second image cannot be flipped with correct meaning.
Fully automated pre-processing might be difficult for this reason.

The mistakes for the fifth image is predicted. I am interested how my model
mistakes and it will discussed in the following part.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           | Prediction                                    |
|:---------------------:|:---------------------------------------------:|
| .60                   | Stop sign                                     |
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |

For the second image ...

| Probability           | Prediction                                    |
|:---------------------:|:---------------------------------------------:|
| .60                   | Stop sign                                     |
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |

For the third image ...

| Probability           | Prediction                                    |
|:---------------------:|:---------------------------------------------:|
| .60                   | Stop sign                                     |
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |

For the fourth image ...

| Probability           | Prediction                                    |
|:---------------------:|:---------------------------------------------:|
| .60                   | Stop sign                                     |
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |

For the fifth image ...

| Probability           | Prediction                                    |
|:---------------------:|:---------------------------------------------:|
| .60                   | Stop sign                                     |
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |

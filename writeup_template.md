# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./modelparam.PNG "Model Visualization"
[image2]: ./center.PNG "Center Driving Image"
[image3]: ./center_flip.PNG "Center Flip Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md for summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file in the submission contains the code of the convolation neural network used for training purpose. It shows how the model was trained and validated on the dataset. The file contains comments that gives insights to different segments of the code. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is inspired from the Nvidia's End to End Learning for Self-Driving Cars Model.
The modified model had has 5 convolution and 3 dense layers, Filters of 5X5 and 3X3 have been used in convolution layers with depth ranging from 24 to 64.
The model uses Relu activation technique to introduce non-linearity in the model. The data is normalised using Keras lambda layer and cropped to appropriate segment using Cropping layer.

#### 2. Attempts to reduce overfitting in the model

The model had dropout layers after each dense layer to reduce overfitting in the model(model.py lines 59 and 61) with keep prob of 0.4 and 0.3 respectively.
The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 64).

#### 4. Appropriate training data

For training I have used Udacity's Sample data which is data of center lane driving along with images from left and right camera's.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I approached towards solution by taking the Nvidia's End to End Learning for Self-Driving Cars Model. Using this model gives a strong base for further processing.

After training on the above model i saw that the model was overfitting,it was resolved by adding Dropout layers after each dense layer of the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle was moving towards the left side more i.e. network was getting biased to the left side , to resolve this i added more images in dataet by flipping thealready available images with a probability of 0.5

Also their were situations where the vehicle was not able to recover from the sides, to tackle this, while adding the image to dataset the image was randomly selected from the 3 available angles i.e left, right or center with the left and right camera's steering angle adjusted by corrective measure of +0.25/-0.25. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 
Layer 0 : Normalization to range -1 to 1.
Layer 1 : Cropping with top crop as 70 and bottom crop as 25
Layer 2 : Convolution with filter shape (24, 5, 5) ,strides (2, 2), valid padding and with relu activation
Layer 3 : Convolution with filter shape (36, 5, 5) ,strides (2, 2), valid padding and with relu activation
Layer 4 : Convolution with filter shape (48, 5, 5) ,strides (2, 2), valid padding and with relu activation
Layer 5 : Convolution with filter shape (64, 5, 5) , valid padding and with relu activation
Layer 6 : Convolution with filter shape (64, 5, 5) , valid padding and with relu activation
Layer 7 : Flatten Layer
Layer 8 : Fully Connected with 512 outputs, relu activation and dropout of 0.5
Layer 9 : Fully Connected with 100 outputs, relu activation and dropout of 0.3
Layer 10 : Fully Connected with 1 outputs and dropout

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

For training Udacity sample training dataset is used which has data of contain center lane driving. Here is an example of the image with center lane driving.

![alt text][image2]

As the First track in simulation is left biased so to prevent the network from being biased towards left angle, I also flipped images and angles so that we can maintain general data for both left and right angle. For example, here is an image that has then been flipped:

![alt text][image3]

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as after 3 epochs the loss was low and moreover the validation loss saturated and their was not much difference between validation and training dataset. I used an adam optimizer so that manually training the learning rate wasn't necessary.

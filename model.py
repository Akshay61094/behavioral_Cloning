import csv
import cv2
from sklearn.utils import shuffle
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D
import random


lines = []
index = 0
# read csv file
with open('data/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for i in reader:
        if index !=0:
            lines.append(i)
        index =1
images = []
measures = []
# load dataset with images from left center or right camera along with some flipped images
for line in lines:
    #select image randomly from left center and right camera
    select = random.choice([0,1,2])
    src = line[select]
    filename = src.split('/')[-1]
    curr_path = 'data/IMG/' + filename
    image = cv2.imread(curr_path)
    #adding corrective measure of +0.25 for left and -0.25 for right image's angle.
    if select == 1:
        angle = float(line[3]) +0.25
    elif select == 2:
        angle = float(line[3]) -0.25
    else:
        angle = float(line[3])
    images.append(image)
    measures.append(angle)
    # Load flipped images with a prob of .5
    if random.random() > 0.5:
        images.append(cv2.flip(image,1))
        measures.append(angle*-1.0)
    
X_train = np.array(images)
y_train = np.array(measures)

print(X_train.shape,y_train.shape)

#Model - > Nvidia Modified
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True, batch_size=32)
model.save('behav.h5')

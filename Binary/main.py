# Libraries imported:

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Train and Test folders defined

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

# Located the Train and Test folders; set size as deisered 

train_dataset = train.flow_from_directory("Train/", 
                                          target_size=(150,150), 
                                          batch_size = 32,
                                          class_mode = 'binary')
                                         
test_dataset = test.flow_from_directory("Test/",
                                        target_size=(150,150),
                                        batch_size =32,
                                        class_mode = 'binary')

# Begin: keras, The Sequential Funtion!

model = keras.Sequential()

# Layer created with 32 filters, 5x5 kernel size, 150x150 resolution and 2 dimensions

model.add(keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(150,150,2)))
model.add(keras.layers.MaxPool2D(2,2))

# Output layer with single neuron which gives 0 for Cat or 1 for Dog 
#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(1,activation='sigmoid'))
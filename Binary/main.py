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

train = ImageDataGenerator(rescale = 1/255)
test = ImageDataGenerator(rescale = 1/255)

# Located the Train and Test folders; set size as deisered 

train_dataset = train.flow_from_directory("/Train/", 
                                          target_size=(150,150), 
                                          batch_size = 32,
                                          class_mode = 'binary')
                                         
test_dataset = test.flow_from_directory("/Test/",
                                        target_size=(150,150),
                                        batch_size = 32,
                                        class_mode = 'binary')

# Begin: keras

model = keras.Sequential()

# Layer created with 32 filters, 5x5 kernel size, 150x150 resolution and 2 dimensions

model.add(keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(150,150,2)))
model.add(keras.layers.MaxPool2D(2,2))

# Sigmod is used here to diffrenciate between 0's and 1's

model.add(keras.layers.Dense(1,activation='sigmoid'))

# Compiling the AI

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Training the model: 

model.fit_generator(train_dataset,
                    steps_per_epoch = 0.5,
                    epochs = 16,
                    validation_data = test_dataset)

                        
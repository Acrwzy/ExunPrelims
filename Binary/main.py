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

model.add(keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(150,150,3)))
model.add(keras.layers.MaxPool2D(2,2))

# 512 neuron layer

model.add(keras.layers.Dense(512,activation='relu'))

# Sigmod is used here to diffrenciate between 0's and 1's

model.add(keras.layers.Dense(1,activation='sigmoid'))

# Compiling the AI

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model: 

model.fit_generator(train_dataset,
                    steps_per_epoch = 0.5,
                    epochs = 16,
                    validation_data = test_dataset)

# Figuring whether 0's or 1's, in an output form

def predictImage(filename):
    img1 = image.load_img(filename,target_size=(150,150))

    plt.imshow(img1)

    Y = image.img_to_array(img1)
    X = np.expand_dims(Y,axis=0)
    
    val = model.predict(X)
    print(val)
    
    if val == 1:
        plt.xlabel("1",fontsize=30)
    elif val == 0:
        plt.xlabel("0",fontsize=30)

                        
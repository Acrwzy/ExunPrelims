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

model = tf.keras.models.Sequential()

# This is the first convolution

tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
tf.keras.layers.MaxPooling2D(2, 2),

# The second convolution

tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

# The third convolution

tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

# The fourth convolution

tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

# The fifth convolution

tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),

# Flatten the results to feed into a DNN

tf.keras.layers.Flatten(),

# 512 neuron hidden layer

tf.keras.layers.Dense(512, activation='relu'),

# Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')

tf.keras.layers.Dense(1, activation='sigmoid')

# Compiling the AI

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model: 

model.fit_generator(train_dataset,
                    steps_per_epoch = 8,
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

                        
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

# located the Train and Test folders; set size as deisered 

train_dataset = train.flow_from_directory("train/", 
                                          target_size=(28,28), 
                                          batch_size = 32,
                                          class_mode = 'binary')
                                         
test_dataset = test.flow_from_directory("Test/",
                                        target_size=(28,28),
                                        batch_size =32,
                                        class_mode = 'binary')


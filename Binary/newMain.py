

import numpy as np 
import pandas as pd  
import os 
from pathlib import Path 
import glob
import seaborn as sns 
import matplotlib.pyplot as plt 
import tensorflow as tf  
from tensorflow.keras import layers
from tensorflow.keras import Model  
from tensorflow.keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator

train_dir = '/Train/'
test_dir = '/Test/'

zero_samples = '/zero/'
one_samples = '/one/'

def make_csv_with_image_labels(zero_samples, one_samples):
    zero_images = zero_samples
    one_images = one_samples

    df = [] 

    for i in zero_images: 
        df.append((i, 0)) 

    for j in one_images: 
        df.append((i, 1)) 

    df = pd.DataFrame(df, columns=["image_path", "label"], index = None) 

    df = df.sample(frac = 1).reset_index(drop=True) 

    return df  

train_csv = make_csv_with_image_labels(zero_samples, one_samples) 

train_csv.head()

len_zero = len(train_csv["label"][train_csv.label == 0])
len_one = len(train_csv["label"][train_csv.label == 1])
arr = np.array([len_zero , len_one]) 
labels = ['ZERO', 'ONE'] 
print("Total No. Of Zero Samples :- ", len_zero)
print("Total No. Of One Samples :- ", len_one)
plt.pie(arr, labels=labels, explode = [0.2,0.0] , shadow=True) 
plt.show()

def get_train_generator(train_dir, batch_size=64, target_size=(224, 224)):  
    
    train_datagen = ImageDataGenerator(rescale = 1./255., 
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 
    train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    batch_size = batch_size, 
                                                    color_mode='rgb',
                                                    class_mode = 'binary')  
    return train_generator  
train_generator = get_train_generator(train_dir)

def get_testgenerator(test_dir,batch_size=64, target_size=(224,224)): 

    test_datagen = ImageDataGenerator( rescale = 1.0/255. )
    test_generator  =  test_datagen.flow_from_directory(test_dir, 
                                                          batch_size  = batch_size, 
                                                          color_mode='rgb',
                                                          class_mode  = 'binary') 
    return test_generator
test_generator = get_testgenerator(test_dir)

model = tf.keras.Sequential([
    layers.Conv2D(64, (3,3), strides=(2,2),padding='same',input_shape= (224,224,3),activation = 'relu'), 
    layers.MaxPool2D(2,2), 
    layers.Conv2D(128, (3,3), strides=(2,2),padding='same',activation = 'relu'),
    layers.MaxPool2D(2,2), 
    layers.Conv2D(256, (3,3), strides=(2,2),padding='same',activation = 'relu'), 
    layers.MaxPool2D(2,2),  
    layers.Flatten(), 
    layers.Dense(158, activation ='relu'), 
    layers.Dense(256, activation = 'relu'), 
    layers.Dense(128, activation = 'relu'), 
    layers.Dense(1, activation = 'sigmoid'), 
]) 
model.summary()

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(train_generator,
                              epochs=15,
                              verbose=1)

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')

model.save('my_model.h5') 

new_model = tf.keras.models.load_model('./my_model.h5') 
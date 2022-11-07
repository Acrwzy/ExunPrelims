

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

def make_csv_with_image_labels(zero_path, one_path):
    zero_images = zero_path.glob('*.jpg')
    one_images = one_path.glob('*.jpg')

    df = [] 

    for i in zero_images: 
        df.append((i, 0)) 

    for j in one_images: 
        df.append((i, 1)) 

    df = pd.DataFrame(df, columns=["image_path", "label"], index = None) 

    df = df.sample(frac = 1).reset_index(drop=True) 

    return df  

train_csv = make_csv_with_image_labels(zero_samples_dir_train,one_samples_dir_train) 

train_csv.head()

len_zero = len(train_csv["label"][train_csv.label == 0])
len_one = len(train_csv["label"][train_csv.label == 1])
arr = np.array([len_zero , len_one]) 
labels = ['ZERO', 'ONE'] 
print("Total No. Of CAT Samples :- ", len_zero)
print("Total No. Of DOG Samples :- ", len_one)
plt.pie(arr, labels=labels, explode = [0.2,0.0] , shadow=True) 
plt.show()
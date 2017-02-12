import os
import pandas as pd
import cv2
import math
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Lambda, Convolution2D, Flatten, Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import json


dir_path = r'C:/VUdacity/data/data/'
new_rows = 66 
new_cols = 200 
angle_shift = 0.12 # tried .15, .20 , .10
test_image_path = r'C:/VUdacity/data/data/IMG/center_2016_12_01_13_31_14_702.jpg'

# crop numbers, tried a few combinations, these seem to work better
c_row_start = 60
c_row_end = 140
c_col_start = 40
c_col_end = 280


def get_recorded_data(csv_path):

    # Reads the driving log csv using pandas
    # Returns panda dataframe
    fieldnames = ['center', 'left', 'right', 'steering', 'throttle']
    types = {'center': str, 'left': str, 'right': str, 'steering': np.float32, 'throttle': np.float32}
        
    result = pd.read_csv(csv_path, names=fieldnames, header=0, usecols=[0, 1, 2, 3, 4], dtype=types)

    return result

def crop_resize_image(image):
    
    # crop   
    cropped_image = image[c_row_start:c_row_end, c_col_start:c_col_end]
   
    # resize
    resized_image = cv2.resize(cropped_image,(new_cols,new_rows))
   
    return resized_image

def transform_image(loaded_img):

    # Ref wiki (https://en.wikipedia.org/wiki/YUV), the best way to create shades of colors seems to be to use the YUV color space.
    # It encodes color image by taking human perception into account.
    # Y portion describes the brightness and UV portion defines the color
    img_array = img_to_array(loaded_img)
    return cv2.cvtColor(img_array,cv2.COLOR_RGB2YUV) # Courtesy of Thomas Antony

def pre_process_image(image_path):

    # load the image
    loaded_image = load_img(image_path)

    # transform
    transformed_image = transform_image(loaded_image)

    # crop and resize
    cropped_image = crop_resize_image(transformed_image)

    return cropped_image


def gen_image_data(csv_data_frame):    

    # shuffle again
    csv_data_frame = csv_data_frame.sample(frac=1).reset_index(drop=True)

    for index, row in csv_data_frame.iterrows():
        
        steering =  row['steering']      
        
        # preprocess each image (center, left & right)
        center_img = pre_process_image(dir_path + row['center'].strip())
        left_img = pre_process_image(dir_path + row['left'].strip())
        right_img = pre_process_image(dir_path + row['right'].strip())

        yield center_img, steering
        yield left_img, steering + angle_shift
        yield right_img, steering - angle_shift


def gen_batch(csv_data_frame, batch_size=32):
    num_rows = len(csv_data_frame.index)
    images_batch = np.zeros((batch_size, new_rows, new_cols, 3))
    steering_batch = np.zeros(batch_size)
    images = gen_image_data(csv_data_frame)
    while 1:
        for i in range(batch_size):            
            try:                               
                images_batch[i], steering_batch[i] =next(images)
            except StopIteration:                
                images = gen_image_data(csv_data_frame)
                images_batch[i], steering_batch[i] = next(images)
          
        yield images_batch, steering_batch


def get_model():

    # Courtesy Thomas Antony
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(new_rows, new_cols, 3), name='Norm1'))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu', name='Conv1'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu', name='Conv2'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu', name='Conv3'))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv4'))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='Conv5'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu', name='FC1'))
    # model.add(Dropout(0.20))
    model.add(Dense(100, activation='relu', name='FC2'))
    # model.add(Dropout(0.20))
    model.add(Dense(50, activation='relu', name='FC3'))
    # model.add(Dropout(0.20))
    model.add(Dense(10, activation='relu', name='FC4'))
    # model.add(Dropout(0.20))
    model.add(Dense(1, name='output'))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    return model

# Read the data log into pandas dataframe
csv_data = get_recorded_data(dir_path + 'driving_log.csv')

# shuffle
csv_data = shuffle(csv_data)

# split to training and validation data
csv_data_train, csv_data_val = train_test_split(csv_data, test_size=0.20,random_state=772288)

batch_size = 32 # tried 64 & 124 
val_samples = len(csv_data_val)

# get the training data generator
train_data_generator = gen_batch(csv_data_train, batch_size)

# validation data generator
val_data_generator = gen_batch(csv_data_val, batch_size)

model = get_model()
model.summary()

history = model.fit_generator(train_data_generator,samples_per_epoch=28000, nb_epoch=2,validation_data=val_data_generator, nb_val_samples=val_samples )

print('History:', history.history)

print("Saving model weights and configuration file.")

model.save_weights('model.h5', True)

with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
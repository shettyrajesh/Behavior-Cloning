# Load libraries
import os
import cv2
import pandas as pd
import numpy as np

from random import randint, uniform
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

drive_array = pd.read_csv('./data/driving_log.csv')

def preprocessImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[60:140, 40:280]
    return cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)

def flip(img, steer):
    rand = randint(0, 2)
    if rand == 0:
        img, steer = cv2.flip(img, 1), -steer
    return img, steer

def brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:,:,2] = img[:,:,2] * np.random.uniform(0.5, 1.5)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def shift(img, steer):
    tr_x = np.random.uniform(-25, 25)
    steer = steer + tr_x/10000*1.6
    tr_y = np.random.uniform(-10, 10)
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    img= cv2.warpAffine(img, Trans_M, (200, 66))
    return img, steer

def gen_image(data):
    cam = np.random.randint(3)
    index = randint(0, len(data) - 1)
    # print("data['left']")
    # print(data['left'])
    # print("data['left'][index]")
    # print(data['left'][index])
    # print("os.path.basename(data['left'][index])")
    # print(os.path.basename(data['left'][index]))

    if cam == 0:
        img = cv2.imread('./data/IMG/' + os.path.basename(data['left'][index]))
        steer = data['steering'][index] + .25
    elif cam == 1:
        img = cv2.imread('./data/IMG/' + os.path.basename(data['center'][index]))
        steer = data['steering'][index]
    else:
        img = cv2.imread('./data/IMG/' + os.path.basename(data['right'][index]))
        steer = data['steering'][index] - .25

    img = preprocessImg(img)
    img = brightness(img)
    img, steer = flip(img, steer)
    img, steer = shift(img, steer)
    return img, steer

# Batch generator
def gen_batch(data, batch_size):
    x_batch = np.zeros((batch_size, 66, 200, 3))
    y_batch = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            x, y = gen_image(data)
            x_batch[i], y_batch[i] = x, y
        yield x_batch, y_batch

model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(66, 200, 3)))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same", init='he_normal'))
model.add(ELU())
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(1164, init='he_normal'))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(100, init='he_normal'))
model.add(ELU())
model.add(Dense(50, init='he_normal'))
model.add(ELU())
model.add(Dense(10, init='he_normal'))
model.add(ELU())
model.add(Dense(1, init='he_normal'))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer="adam", loss="mse")


# Split data into train and validation sets
train, validation = train_test_split(drive_array, test_size=0.10, random_state=1111)
train = train.reset_index()
validation = validation.reset_index()

model.summary()
# Train the model
model.fit_generator(gen_batch(train, 128),samples_per_epoch = 32000,nb_epoch = 7,validation_data = gen_batch(validation, 128),nb_val_samples = 3200, verbose = 1)

model.save('model.h5')
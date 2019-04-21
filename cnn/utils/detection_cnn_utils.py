import keras
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Input, Dropout, Flatten, Activation, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import json
import pickle
from utils.preprocess_utils import *

SHAPE = (64, 64)
CHANNEL = 3
MAX_DIGIT_DETECTED = 4
train_folders = 'data/train'
test_folders = 'data/test'
train_data_name = 'data/train_data.pkl'
test_data_name = 'data/test_data.pkl'


def train_detection_cnn():
    with open(train_data_name, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_data_name, 'rb') as f:
        test_data = pickle.load(f)
    train_formatter = DataFormatter(train_data, train_folders, SHAPE, channel=CHANNEL, add_neg=False)
    train_data = train_formatter.get_formatted_data_for_modeling()
    test_formatter = DataFormatter(test_data, test_folders, SHAPE, channel=CHANNEL, add_neg=False)
    test_data = test_formatter.get_formatted_data_for_modeling()

    train_img_size = (64, 64)
    train_images, train_labels = get_detecting_imgages_labels(train_formatter, train_img_size)
    test_images, test_labels = get_detecting_imgages_labels(test_formatter, train_img_size)
    model = create_detection_cnn()
    model.compile(loss='mse', optimizer='adadelta')
    model_checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/detection_model.hdf5',
                                                         monitor='loss',
                                                         save_best_only=True,
                                                         verbose=2)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                  factor=0.1,
                                                  verbose=1,
                                                  patience=4,
                                                  cooldown=1,
                                                  min_lr=0.0001)
    callback = [model_checkpointer, reduce_lr]
    model.fit(train_images, train_labels,
              epochs=50, batch_size=128,
              validation_split=0.2, verbose=1,
               callbacks=callback
             )


def create_detection_cnn():
    k_size = 3
    cnn1 = Input(shape=(64, 64, 1))
    # x = BatchNormalization()(cnn1)
    x = Conv2D(32, k_size, k_size, activation='relu', border_mode='same')(cnn1)
    x = BatchNormalization()(x)
    x = Conv2D(32, k_size, k_size, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, k_size, k_size, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, k_size, k_size, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, k_size, k_size, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, k_size, k_size, activation='relu', border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    h = Dense(4)(x)

    model = Model(input=cnn1, output=h)
    return model
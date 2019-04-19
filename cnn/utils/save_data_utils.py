import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import random

from utils.preprocess_utils import *

train_folders = 'data/train'
test_folders = 'data/test'
train_data_name = 'data/train_data.pkl'
test_data_name = 'data/test_data.pkl'


def process_and_save_mat_info():
    train_mat = os.path.join(train_folders, 'digitStruct.mat')
    train_data = get_data_info(train_mat)

    test_mat = os.path.join(test_folders, 'digitStruct.mat')
    test_data = get_data_info(test_mat)

    with open(train_data_name, 'wb') as f:
        pickle.dump(train_data, f)

    with open(test_data_name, 'wb') as f:
        pickle.dump(test_data, f)


process_and_save_mat_info()

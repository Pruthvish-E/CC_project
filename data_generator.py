#imports
import os
import numpy as np
np.random.seed(1969)
import tensorflow as tf
tf.set_random_seed(1969)
from numpy import genfromtxt

from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile


train_data_path = os.path.join('/media/user/ProfSavitha/Speech_Recognition/data-telugu/train-mfsc')
test_data_path = os.path.join('/media/user/ProfSavitha/Speech_Recognition/data-telugu/test-mfsc')
val_data_path = os.path.join('/media/user/ProfSavitha/Speech_Recognition/data-telugu/val-mfsc')

class DataGenerator():

    def list_npy_fname(self, dirpath, heirarchy, ext='npy',restrictA=False):
        aCount=0
        fpaths = sorted(glob(os.path.join(dirpath, r'*/*' + ext)))
        labels = []
        X = []
        for fpath in fpaths:
            split_by_slash = fpath.split('/')
            speaker = split_by_slash[-2]
            file_name = split_by_slash[-1]
            file_name_split = file_name.split('-')
            syllable = file_name_split[1]
            label = file_name_split[1+heirarchy]
            label = label[:label.find('.')]
            '''if label=='A' and restrictA:
                if aCount>2500:
                    continue
                else:
                    aCount+=1'''
            labels.append(label)
            X.append(genfromtxt(fpath))
        print("There are %d samples"%len(labels))
        print("THe maximum length of the code is ")

    def __init__(self, heirarchy, path):
        self.X, self.y = self.list_npy_fname(path, heirarchy)
        self.progress = 0
        self.n_samples = self.X.shape[0]
    
    def get_next_batch(batch_size):
        while True:
            pass
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


class DataGenerator():

    def list_npy_fname(self, dirpath, heirarchy, ext='npy',restrictA=False):
        aCount=0
        fpaths = sorted(glob(os.path.join(dirpath, r'*/*' + ext)))
        self.labels = []
        self.X = []
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

            self.labels.append(label)
            self.X.append(genfromtxt(fpath))
        print("There are %d samples"%len(labels))
        
        #padding section of hte code 
        self.maxlen = max([x.shape[0] for x in self.X])
        self.features = self.X[0].shape[1]
        print("The maximum length of the dataset is %d, padding to this length"%self.maxlen)
        X_new = []

        for i in self.X:
            listx = list(i)
            for i in range(maxlen - len(listx)):
                listx.append(np.zeros((self.features)))
            X_new.append(listx)
        self.X, X_new = X_new, self.X

        del X_new
        self.X = np.asarray(self.X, dtype = np.float32)

        #convert y to onehot encoded vectors
        self.n_classes = len(set(self.labels))
        self.class_list = list(set(self.labels)).sort()
        self.y = []
        print("There are %s classes. Their distribution:"%self.n_classes)
        for i in set(self.labels):
            print("%s : %d"%(i, self.labels.count(i)))

        for i in self.labels:
            onehot = [0 for k in range(len(labels))]
            onehot[self.labels.index(i)] = 1
            self.y.append(onehot)

        self.y = np.asarray(self.y, dtype = np.float32)

    def __init__(self, heirarchy, path):
        self.list_npy_fname(path, heirarchy)
        self.progress = 0
        self.n_samples = self.X.shape[0]
    
    def get_next_batch(batch_size):
        while True:
            pass

    def getdata(self):
        return np.reshape(self.X, (-1, self.maxlen, self.features, 1)), self.y
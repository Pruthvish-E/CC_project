#imports
import os
import numpy as np
np.random.seed(1969)
from numpy import genfromtxt

from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile
import pickle
with open('dict.pickle', 'rb') as handle:
    heirarchydict = pickle.load(handle)
class DataGenerator():

    def list_npy_fname(self, dirpath, heirarchy, ext='npy',restrictA=False):
        self.labels = []
        self.X = []
        fpaths = sorted(glob(os.path.join(self.path, r'*/*' + ext)))
        for fpath in fpaths:
            split_by_slash = fpath.split('/')
            speaker = split_by_slash[-2]
            file_name = split_by_slash[-1]
            file_name_split = file_name.split('-')
            syllable = file_name_split[1]
            label = file_name_split[1+heirarchy]
            if(heirarchy == 2):
                label = label[:label.find('.')]

            self.labels.append(heirarchydict[heirarchy][label].index(1))
            self.X.append(genfromtxt(fpath))
            
        print("There are %d samples"%len(self.labels))
        
        #padding section of hte code 
        self.maxlen = max([x.shape[0] for x in self.X] + [self.maxlen])
        self.features = self.X[0].shape[1]
        print("The maximum length of the dataset is %d, padding to this length"%self.maxlen)
        X_new = []

        for i in self.X:
            listx = list(i)
            for i in range(self.maxlen - len(listx)):
                listx.append(np.zeros((self.features)))
            X_new.append(listx)
        self.X, X_new = X_new, self.X

        del X_new
        self.X = np.asarray(self.X, dtype = np.float32)
        self.y = np.asarray(self.labels, dtype = np.int32)
        del self.labels
        

    def __init__(self, heirarchy, path, maxlen = float('-inf')):
        self.maxlen = maxlen
        self.path = path
        self.list_npy_fname(path, heirarchy)
        self.progress = 0
        self.n_samples = self.X.shape[0]
    def get_next_batch(self, batch_size):
        while True:
            pass

    def getdata(self):
        return np.reshape(self.X, (-1, self.maxlen, self.features, 1)), self.y

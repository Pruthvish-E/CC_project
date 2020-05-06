#imports
import os
import numpy as np
np.random.seed(1969)
from numpy import genfromtxt
from tqdm import tqdm

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
    def list_npy_fname(self, dirpath, heirarchy,ext='npy',restrictA=False):
        label_counts = {}
        self.labels = []
        self.X = None
        self.count = 0
        fpaths = sorted(glob(os.path.join(self.path, r'*/*' + ext)))
        for fpath in tqdm(fpaths):
            split_by_slash = fpath.split('/')
            speaker = split_by_slash[-2]
            file_name = split_by_slash[-1]
            file_name_split = file_name.split('-')
            syllable = file_name_split[1]
            label = file_name_split[1+heirarchy]
            if(heirarchy == 2):
                label = label[:label.find('.')]
            
            newlabel = heirarchydict[heirarchy][label].index(1)
            
            try:
                label_counts[newlabel] += 1;
            except:
                label_counts[newlabel] = 1;

            if(label_counts[newlabel] == 32120):
                continue;
                
            sample = genfromtxt(fpath)
            if(sample.shape[0]<= self.maxlen):
                if self.X is None:
                    self.X = np.zeros((32120*6, self.maxlen, sample.shape[1]), dtype = np.float32)
                self.X[self.count,:sample.shape[0],:sample.shape[1]]= sample
                self.count+=1
                self.labels.append(heirarchydict[heirarchy][label].index(1))

        print("There are %d samples"%len(self.labels))
        self.y = np.asarray(self.labels, dtype = np.int32)
        del self.labels
        gc.collect()

    def __init__(self, heirarchy, path, maxlen):
        self.maxlen = maxlen
        self.path = path
        self.list_npy_fname(path, heirarchy)
        self.progress = 0
        self.heirarchy = heirarchy
        self.n_classes = len(heirarchydict[heirarchy])
        self.n_samples = self.y.shape[0]
        
    def get_class_weights(self):
        total = self.y.shape[0]
        return {heirarchydict[self.heirarchy][key].index(1) : total//np.count_nonzero(self.y == key) for key in heirarchydict[self.heirarchy]}
    def get_next_batch(self, batch_size):
        while True:
            pass

    def getdata(self):
#         return np.reshape(self.X, (-1, self.maxlen, self.features, 1)), self.y
        return self.X[:self.count], self.y


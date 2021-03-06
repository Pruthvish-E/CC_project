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

with open('freqdict.pickle', 'rb') as handle:
    freqdict = pickle.load(handle)
class DataGenerator():
    def __init__(self, heirarchy, path,batch_size, maxlen = float('-inf'),  prefetch = 50):
        self.maxlen = maxlen
        self.path = path
        self.heirarchy = heirarchy
        self.prefetch = prefetch
        self.batch_size = batch_size
        self.batches = []
        self.n_classes = len(freqdict[heirarchy].keys())
        self.datagen = self.get_next_sample()
        self.add_batches()
        
    def get_class_weights(self):
        total = sum(list(freqdict[self.heirarchy].values()))
        return {heirarchydict[self.heirarchy][key].index(1) : total//freqdict[self.heirarchy][key] for key in freqdict[self.heirarchy]}

    def get_next_sample(self, ext='npy'):
        fpaths = sorted(glob(os.path.join(self.path, r'*/*' + ext)))
        while(True):
            for fpath in fpaths:
                split_by_slash = fpath.split('/')
                speaker = split_by_slash[-2]
                file_name = split_by_slash[-1]
                file_name_split = file_name.split('-')
                syllable = file_name_split[1]
                label = file_name_split[1+self.heirarchy]
                if(self.heirarchy==2):
                    label = label[:label.find('.')]  
                listx = list(genfromtxt(fpath))
                self.features = listx[0].shape[0]
                for i in range(self.maxlen - len(listx)):
                    listx.append(np.zeros((self.features)))
                yield listx, heirarchydict[self.heirarchy][label].index(1)
                
    def add_batches(self):
        for i in range(self.prefetch):
            batchx = []
            batchy = []
            for i in range(self.batch_size):
                d = next(self.datagen)
                batchx.append(d[0])
                batchy.append(d[1])
            self.batches.append((np.asarray(batchx, dtype = np.float32), np.asarray(batchy, dtype = np.int32)))
   
        
    def get_next_batch(self):
        while True:
            while len(self.batches)!=0:
                batch = self.batches.pop(0)
                x = batch[0]
#                 x = np.expand_dims(x, 3) #adding the last dimension (bs, l, w, "1")
                yield x, batch[1]
            self.add_batches()
            

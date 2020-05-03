import os
import numpy as np
from numpy import genfromtxt

from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile
import pickle


def list_npy_fname(self, dirpath,  ext='npy'):
        lens = []
        fpaths = sorted(glob(os.path.join(dirpath, r'*/*' + ext)))
        for fpath in fpaths:
           x = genfromtxt(fpath)
           lens.append(x.shape[0])

        return lens
            
        
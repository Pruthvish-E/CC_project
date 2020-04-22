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

from keras import optimizers, losses, activations, models
from keras.callbacks import TensorBoard
from keras.models import Sequential
from tqdm import tqdm
#from sklearn.model_selection import GroupKFold

from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

hierarchy = 1
max_frames = 413
num_features_per_frame = 27

L = 413
legal_labels = 'A E I O U X'.split()

train_data_path = os.path.join('/media/user/ProfSavitha/Speech_Recognition/data-telugu/train-mfsc')
test_data_path = os.path.join('/media/user/ProfSavitha/Speech_Recognition/data-telugu/test-mfsc')
val_data_path = os.path.join('/media/user/ProfSavitha/Speech_Recognition/data-telugu/val-mfsc')

tensorboard_log_dir = 'logs'
classifier_save_dir = '../classifiers'
prob_dir = 'probs'
model_name = ''

def list_npy_fname(dirpath, ext='npy',restrictA=False):
    #print(dirpath)
    aCount=0
    fpaths = sorted(glob(os.path.join(dirpath, r'*/*' + ext)))
    speakers = []
    labels = []
    fnames = []
    for fpath in fpaths:
        split_by_slash = fpath.split('/')
        speaker = split_by_slash[-2]
        file_name = split_by_slash[-1]
        #file_name = file_name[:file_name.find('.')]
        file_name_split = file_name.split('-')
        syllable = file_name_split[1]
        label = file_name_split[1+hierarchy]
        label = label[:label.find('.')]
        if label=='A' and restrictA:
            if aCount>2500:
                continue
            else:
                aCount+=1
        #print('speaker:'+speaker+' file:'+file_name)
        #print('\tlabel:'+label)
        speakers.append(speaker)
        labels.append(label)
        fnames.append(file_name)

    return speakers,labels, fnames


def pad_zero(samples):
    return np.pad(samples,pad_width=((L-len(samples),0),(0,0)), mode='constant' , constant_values=(0,0))

def label_transform(labels):
    print('Generating hotbits')
    return pd.get_dummies(pd.Series(labels))


train_speakers, train_labels, train_fnames = list_npy_fname(train_data_path)
print("A=",str(train_labels.count('A')))
print("E=",str(train_labels.count('E')))
print("I=",str(train_labels.count('I')))
print("O=",str(train_labels.count('O')))
print("U=",str(train_labels.count('U')))
print("X=",str(train_labels.count('X')))

num_train_samples = len(train_fnames)
print("No of train samples:",num_train_samples)
y_train = []
x_train = np.zeros((num_train_samples,max_frames,num_features_per_frame,1),np.float32)
ix = 0


print("\nEntering train data creation loop")
for speaker, label, fname in tqdm(zip(train_speakers, train_labels, train_fnames)):
    samples = genfromtxt(os.path.join(train_data_path, speaker, fname))
    samples = pad_zero(samples)
    samples = np.reshape(samples,(max_frames,num_features_per_frame,1))
    x_train[ix,:,:,:] = samples
    y_train.append(label)
    ix += 1

y_train = label_transform(y_train)
train_label_index = y_train.columns.values
print('Train label Index',train_label_index)
y_train = y_train.values
y_train = np.array(y_train)


val_speakers, val_labels, val_fnames = list_npy_fname(val_data_path)
num_val_samples = len(val_fnames)
y_val = []
x_val = np.zeros((num_val_samples,max_frames,num_features_per_frame,1),np.float32)
ix = 0
print("\nStarting val array build")
for speaker, label, fname in tqdm(zip(val_speakers, val_labels, val_fnames)):
    samples = genfromtxt(os.path.join(val_data_path, speaker, fname))
    samples = pad_zero(samples)
    samples = np.reshape(samples,(max_frames,num_features_per_frame,1))
    x_val[ix,:,:,:] = samples
    y_val.append(label)
    ix += 1

y_val= label_transform(y_val)
val_label_index = y_val.columns.values
print('Val label index',val_label_index)
y_val = y_val.values
y_val = np.array(y_val)

del train_speakers, train_labels, train_fnames
gc.collect()


classifier = Sequential()
classifier.add(Conv2D(32,(3,3), input_shape=(max_frames,num_features_per_frame,1), activation ='relu'))
classifier.add(Dropout(0.4))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=6, activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
classifier.summary()
weights = classifier.get_weights()

EPOCHS = 30
BATCH_SIZE = 512


shape = None


val_loss = np.ones((EPOCHS),np.float32)

print("Train shape",x_train.shape,"\n",y_train.shape)
print("Val shape",x_val.shape,"\n",y_val.shape)

classifier.set_weights(weights)
classifier.reset_states()
tensorboard = TensorBoard(log_dir=tensorboard_log_dir+'/train_'+model_name)
history = classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_data=(x_val, y_val), epochs=EPOCHS, shuffle=True, verbose=1, callbacks=[tensorboard])
val_loss[:] = history.history['val_loss']

#val_mean = np.mean(val_loss)
best_loss = np.min(val_loss)
best_epoch = np.argmin(val_loss)
print('Best epoch: {} Best loss: {}'.format(best_epoch,best_loss))
#classifier.set_weights(weights)
#classifier.reset_states()
tensorboard = TensorBoard(log_dir=tensorboard_log_dir+'/'+model_name)
#classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, initial_epoch=best_epoch, shuffle=True, verbose=1, callbacks=[tensorboard])

classifier.save(classifier_save_dir+'/'+model_name+'_{}.h5'.format(best_loss))

print("\nTesting:")
def test_data_generator(batch=32):
    fpaths = sorted(glob(os.path.join(test_data_path, r'*/*'+'npy')))
    i = 0
    for path in fpaths:
       # print(path)
        if i == 0:
            imgs = []
            fnames = []
        i += 1

        sample = genfromtxt(path)
        sample = pad_zero(sample)
        sample = np.reshape(sample,(max_frames,num_features_per_frame,1))
        imgs.append(sample)
        fnames.append(path.split('/')[-1])
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)

        yield fnames, imgs
    raise StopIteration()


gc.collect()

index = []
results = []
probs = []
for fnames, imgs in tqdm(test_data_generator(batch=32)):
    predicts = classifier.predict(imgs)
    probs.extend(predicts)
    predicts = np.argmax(predicts, axis=1)
    predicts = [train_label_index[p] for p in predicts]
    index.extend(fnames)
    results.extend(predicts)

df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = index
df['label'] = results
#df.to_csv(os.path.join(out_path, 'subs/cnn_sub_{}.csv'.format(best_loss)), index=False)
df.to_csv('subs/'+model_name+'_sub_{}.csv'.format(best_loss), index=False)
probs = np.array(probs)
np.save(prob_dir+'/'+model_name+'_probs.npy',probs)

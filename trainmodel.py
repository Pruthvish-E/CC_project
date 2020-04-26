from keras import optimizers, losses, activations, models
from keras.callbacks import TensorBoard
from keras.models import Sequential
from tqdm import tqdm
#from sklearn.model_selection import GroupKFold

from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from data_generator import DataGenerator

heirarchy = 1
train_path = "/home/rohan/data-telugu/train-mfsc"
test_path = "/home/rohan/data-telugu/test-mfsc"
val_path = "/home/rohan/data-telugu/val-mfsc"
BATCH_SIZE = 512
tensorboard_log_dir = 'logs'
classifier_save_dir = '../classifiers'
prob_dir = 'probs'
model_name = ''
EPOCHS = 50

train_data = DataGenerator(heirarchy, train_path)
val_data = DataGenerator(heirarchy, val_path, maxlen=train_data.maxlen)

classifier = Sequential()
classifier.add(Conv2D(32,(3,3), input_shape=(train_data.maxlen,train_data.features,1), activation ='relu'))
classifier.add(Dropout(0.4))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=train_data.n_classes, activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
classifier.summary()

tensorboard = TensorBoard(log_dir=tensorboard_log_dir+'/train_'+model_name)
history = classifier.fit(*train_data.get_data(), batch_size=BATCH_SIZE, validation_data=val_data.getdata(), epochs=EPOCHS, shuffle=True, verbose=1, callbacks=[tensorboard])

classifier.save("test.h5")

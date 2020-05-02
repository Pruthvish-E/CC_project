from keras import optimizers, losses, activations, models
from keras.callbacks import TensorBoard
from keras.models import Sequential
from tqdm import tqdm
import keras
#from sklearn.model_selection import GroupKFold
import old_generator
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from data_generator import DataGenerator

heirarchy = 1
train_path = "train-mfsc"
test_path = "test-mfsc"
val_path = "val-mfsc"
BATCH_SIZE = 512
tensorboard_log_dir = 'logs'
classifier_save_dir = '../classifiers'
prob_dir = 'probs'
model_name = ''
EPOCHS = 10

val_data = old_generator.DataGenerator(heirarchy, val_path, maxlen=413)
train_data = DataGenerator(heirarchy, train_path, BATCH_SIZE, maxlen = 413, prefetch=750)

print("finished loading all the data")
# classifier = Sequential()
# classifier.add(Conv2D(32,(3,3), input_shape=(train_data.maxlen,train_data.features,1), activation ='relu'))
# classifier.add(Dropout(0.4))
# classifier.add(MaxPooling2D(pool_size =(2,2)))
# classifier.add(Flatten())
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(units=train_data.n_classes, activation='softmax'))

# classifier.compile(optimizer='adam',loss=keras.losses.sparse_categorical_crossentropy,metrics=['categorical_accuracy'])
# classifier.summary()

classifier = load_model('test.h5')

class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.curmax = float("-inf")
    def on_epoch_end(self, epoch, logs=None):
        validation_data=val_data.getdata()
        _, acc = classifier.evaluate(*validation_data)
        print("Epoch %d has val acc %f"%(epoch, acc))
        if acc > self.curmax:
            curmax = acc
            classifier.save("max.h5")
        classifier.save("latest.h5")
        return super().on_epoch_end(epoch, logs=logs)

tensorboard = TensorBoard(log_dir=tensorboard_log_dir+'/train_'+model_name)

cb = CustomCallback()
#cp = keras.callbacks.ModelCheckpoint(filepath='/', mode='max', monitor='val_acc', verbose=2, save_best_only=True)
#history = classifier.fit(*train_data.get_data(), batch_size=BATCH_SIZE, validation_data=val_data.getdata(), epochs=EPOCHS, shuffle=True, verbose=1, callbacks=[tensorboard])
history = classifier.fit_generator(train_data.get_next_batch(), steps_per_epoch = (751575//BATCH_SIZE), epochs=EPOCHS, shuffle=True, verbose=1, callbacks=[tensorboard,cb], class_weight = train_data.get_class_weights())

classifier.save("test.h5")

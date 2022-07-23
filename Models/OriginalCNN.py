import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
# LSTM for international airline passengers problem with window regression framing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization
from pandas import read_csv
from keras.layers.convolutional import Conv1D, MaxPooling1D
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Flatten, Dense, Dropout
import random

import warnings
warnings.filterwarnings('ignore')
#from torch import flatten
random.seed(72)
data = read_csv('../dataBinary.csv')
# dataset excluding target attribute
X = data.iloc[:, 0:11]
Y = data[['Label']].values  # target attribute

# splitting the dataset 75% for training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

X_train = X_train.to_numpy()
x_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
model = Sequential()
input_shape = (x_train.shape[1], 1)
model.add(Conv1D(128, kernel_size=3, padding='same',
          activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(128, kernel_size=3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
#model.add(Conv1D(64,kernel_size=3,padding = 'same', activation='relu'))
#model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
model.summary()
adam = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy',
              optimizer=adam, metrics=['accuracy'])

keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=2,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,

)
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./log/OriginalCNN")
# training the model on training dataset
modelHistory = model.fit(x_train, y_train, epochs=400,
              batch_size=256, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test)
#plot_model(model, to_file='org_CNN_plot.png',
#           show_shapes=True, show_layer_names=True)
#dumb(history)
print('Test score:', score)
print('Test accuracy:', acc)

plt.plot(modelHistory.history['loss'], label='train loss')
plt.plot(modelHistory.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(modelHistory.history['accuracy'], label='train acc')
plt.plot(modelHistory.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

print("done")
model.save('OriginalCNN.h5')


import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

# LSTM for international airline passengers problem with window regression framing
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization

import warnings
warnings.filterwarnings('ignore')
#from torch import flatten

data = read_csv('../dataBinary.csv')
# dataset excluding target attribute (encoded, one-hot-encoded,original)
X = data.iloc[:, 0:11]
Y = data[['Label']].values  # target attribute

# splitting the dataset 75% for training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

X_train = X_train.to_numpy()
x_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


lstm = Sequential()
lstm.add(LSTM(512, input_shape=(
    x_train.shape[1], 1), return_sequences=True))
#model.add(Dropout(0.1))
lstm.add(BatchNormalization())
lstm.add(Dropout(0.3))

lstm.add(LSTM(256, activation="relu", return_sequences=True))
lstm.add(BatchNormalization())
lstm.add(Dropout(0.5))


lstm.add(LSTM(128, activation="relu", return_sequences=True))
lstm.add(BatchNormalization())
lstm.add(Dropout(0.3))

lstm.add(LSTM(64, activation="relu", return_sequences=True))
lstm.add(BatchNormalization())
lstm.add(Dropout(0.3))


lstm.add(LSTM(32, activation="relu"))
lstm.add(BatchNormalization())
lstm.add(Dropout(0.2))


lstm.add(Dense(2))
lstm.add(Activation('sigmoid'))
rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
lstm.compile(loss='binary_crossentropy',
             optimizer='rmsprop', metrics=['accuracy'])


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
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./log/LSTM")

# training the model on training dataset
modelHistory = lstm.fit(x_train, y_train, epochs=150,
             batch_size=64, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
score, acc = lstm.evaluate(X_test, y_test)
#dumb(history)
plot_model(lstm, to_file='LSTM_model_plot.png',
           show_shapes=True, show_layer_names=True)
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
lstm.save('LSTM.h5')

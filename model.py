import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
import os
import time
from scipy.stats import reciprocal
from scikeras import wrappers
import pathlib
import matplotlib.pyplot as plt
from functools import partial


# getting current path of the file(model.py)
cur_dir = pathlib.Path(__file__).parent.resolve() 
# creating callback which shows current ratio of validation loss and training loss
class RatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f" Ratio - {logs['val_loss'] / logs['loss']}")

# building model using functional API to use it in randomized/grid search
def build_model(n_neurons, n_hidden, learning_rate, input_shape=[12544]):
    model = keras.models.Sequential()
    options = {'input_shape': input_shape}
    default_conv = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', kernel_initializer='he_normal', input_shape=(112, 112, 1))
    model.add(default_conv(filters=10))
    model.add(keras.layers.MaxPool2D())
    # model.add(default_conv(filters=20))
    # model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Flatten())
    for layer_idx in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, **options))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Activation('swish'))
        options = {}
    model.add(keras.layers.Dense(2000, activation='softmax'))
    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# creating and getting log dir for tensorboard callback
def get_dir():
    log_dir = os.path.join(cur_dir, 'logs')
    run_dir = os.path.join(log_dir, time.strftime('run_%Y-%m-%d %H-%M-%S'))
    return run_dir

# parsing function to load data
def parse(serialized_example):
    feature_desc = {'features': tf.io.VarLenFeature(dtype=tf.string),
                 'labels': tf.io.VarLenFeature(dtype=tf.int64)}
    return tf.io.parse_single_example(serialized_example, feature_desc)


run_log_dir = get_dir()

print(cur_dir)
# loading and splitting dataset
data = tf.data.TFRecordDataset(filenames=os.path.join(cur_dir, 'encoded_data.tfrecord')).map(parse)
# reading data, split it to X and y, currently sampling first 1000 samples to test if the model works properly
# currently thinking how to train the model on all the 144k samples without running out of RAM
for example in data:
    print(example)
    X = tf.sparse.to_dense(example['features']) 
    # for image in X:
    #     print(tf.io.decode_png(image))
    X = tf.convert_to_tensor([tf.io.decode_jpeg(image) for image in X])
    y = tf.sparse.to_dense(example['labels'])


# flatten_layer = keras.layers.Flatten()
# plt.imshow(X[0])
# plt.show()
# X = flatten_layer(X)
# X = rescaler(X)
X = X._numpy()
y = y._numpy()
# for idx, img in enumerate(X):
#     plt.subplot(2, 5, idx+1)
#     plt.imshow(img)
# plt.show()
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75)
print(y_valid, y_train)
# creating callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath='best_model_small_set.keras')
early_stop_cb = keras.callbacks.EarlyStopping(patience=10)
tb_cb = keras.callbacks.TensorBoard(log_dir=run_log_dir)
ratio_cb = RatioCallback()
learning_sch = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
callbacks = [checkpoint_cb, early_stop_cb, ratio_cb, learning_sch]
# creating model wrapper and params for RS
model_wrapper = wrappers.KerasClassifier(model=build_model, n_neurons=10, n_hidden=10, learning_rate=1, input_shape=[12544], callbacks=callbacks)
model_params = {'n_neurons': [i for i in range(2000, 10000, 1000)],
                'n_hidden': [i for i in range(2, 10)],
                'learning_rate': reciprocal(0.0001, 0.01).rvs(10)} 
# since training is really time-consuming I load the model with weights on the step when I stopped training
loaded_model = keras.models.load_model(filepath=os.path.join(cur_dir, 'best_model_small_set.keras'))
loaded_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=20, batch_size=512, callbacks=callbacks)
# random_search = GridSearchCV(model_wrapper, model_params, cv=2, verbose=2)
# random_search.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=20, batch_size=512)
print(loaded_model.estimate(X_test, y_test))

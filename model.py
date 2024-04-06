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
from sklearn.decomposition import PCA


lrn = keras.layers.Lambda(tf.nn.local_response_normalization, arguments={'depth_radius': 2, 'alpha': 0.0002, 'beta': 0.75})
DefaultConv = partial(keras.layers.Conv2D, kernel_size=3, strides=1, activation='relu', kernel_initializer='he_normal', use_bias=False, 
                    padding='same')
# creating callback which shows current ratio of validation loss and training loss
class RatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f" Ratio - {logs['val_loss'] / logs['loss']}")

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__( **kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv(filters=filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv(filters=filters),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv(filters=filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()
            ]
        
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        # print(skip_Z)
        return self.activation(Z + skip_Z)
    

# building model to use it in randomized/grid search
def build_model(n_neurons, n_hidden, learning_rate, input_shape=[12544]):
    model = keras.models.Sequential([DefaultConv(filters=8, kernel_size=4, input_shape=[112, 112, 1]),
                                    keras.layers.BatchNormalization(),
                                    DefaultConv(filters=16),
                                    keras.layers.BatchNormalization(),
                                    # DefaultConv(filters=32),
                                    # keras.layers.BatchNormalization(),
                                    keras.layers.MaxPool2D(pool_size=3, strides=2)
                                    ]
)
    options = {'input_shape': input_shape}
    prev_filters = 32
    for filters in [64] * 2 + [128]:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters=filters, strides=strides))
        prev_filters = filters
    
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Flatten())
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

# performing grid/randomized cv search, not really useful for out task because of time consumation so I just use it for fitting only
def search(n_hidden, n_neurons, learning_rate, search_type='g'):
    model_wrapper = wrappers.KerasClassifier(model=build_model, n_neurons=10, n_hidden=10, learning_rate=1, input_shape=[12544], callbacks=callbacks)
    model_params = {'learning_rate': learning_rate, 
                    'n_neurons': n_neurons,
                    'n_hidden': n_hidden
                    } 
    if search_type == 'g':
        search = GridSearchCV(model_wrapper, model_params, cv=2, verbose=4)
    else:
        search = RandomizedSearchCV(model_wrapper, model_params, n_iter=10, cv=2, verbose=4)
    search.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=216)
    print(search.best_estimator_.get_params())

# since the training is really time consuming we to continue training whenever we want
def continue_training(filepath, X_train, y_train, X_valid, y_valid, callbacks, epochs=10, batch_size=2048):
    loaded_model = keras.models.load_model(filepath=filepath, custom_objects={'ResidualUnit': ResidualUnit})
    loaded_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_size, callbacks=callbacks)


def test_pretrained(filepath, X_test, y_test):
    # instead of using custom_objects would be better to decorate the ResidualUnit class
    loaded_model = keras.models.load_model(filepath=filepath, custom_objects={'ResidualUnit': ResidualUnit})
    print(loaded_model.evaluate(X_test, y_test))
     
# loading and splitting dataset
def get_subset(X, y, one_class_samples_size):
    subset_X = []
    subset_y = []
    default_one_class_size = 72
    total_size = X.shape[0]
    cur_size = 0
    while cur_size < total_size:
        # print(cur_size)
        for idx in range(cur_size, min(cur_size + one_class_samples_size, total_size)):
            subset_X.append(X[idx])
            subset_y.append(y[idx])
        cur_size += default_one_class_size
    # print(tf.convert_to_tensor(subset_X), tf.convert_to_tensor(subset_y))
    # print(len(subset_X))
    return tf.convert_to_tensor(subset_X), tf.convert_to_tensor(subset_y)

def read_data(filepath, one_class_samples_size=72):
    data = tf.data.TFRecordDataset(filenames=filepath).map(parse)
    for example in data:
        X = tf.sparse.to_dense(example['features'])
        X = tf.convert_to_tensor([tf.io.decode_jpeg(image) for image in X])
        y = tf.sparse.to_dense(example['labels'])
        X, y = get_subset(X, y, one_class_samples_size)
    return X, y
    
def data_augmentation(X, y):
    X_augmented = keras.layers.RandomRotation(60, seed=0)(X)
    X_augmented = keras.layers.RandomContrast(factor=0.5, seed=0)(X_augmented)
    return X_augmented, y

def preprocess(X, resize=False, rescale=False, reduce_channels=0):
    X_preprocessed = X
    if reduce_channels:
        X_preprocessed = X_preprocessed[..., :reduce_channels]
    if resize:
        X_preprocessed = tf.keras.layers.Resizing(height=112, width=112, crop_to_aspect_ratio=True)(X_preprocessed)
    if rescale:
        rescaler = keras.layers.Rescaling(scale=1./255)
        X_preprocessed = rescaler(X_preprocessed)
    return X_preprocessed

if __name__ == '__main__':
    tf.random.set_seed(42)
    # getting current path of the file(model.py)
    cur_dir = pathlib.Path(__file__).parent.resolve() 
    
    run_log_dir = get_dir()
    X, y = read_data(filepath=os.path.join(cur_dir, 'encoded_data.tfrecord'), one_class_samples_size=30)
    X = preprocess(X, rescale=True)
    print(X.shape)
    X_augmented, y_augmented = data_augmentation(X, y)
    X = tf.concat(values=[X, X_augmented], axis=0)
    y = tf.concat(values=[y, y_augmented], axis=0)
    print(X.shape, y.shape)
    X = X._numpy()
    y = y._numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.75, random_state=0)

    # creating callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath='models/one_channel_without_two_convs_two_ru.keras')
    early_stop_cb = keras.callbacks.EarlyStopping(patience=10)
    tb_cb = keras.callbacks.TensorBoard(log_dir=run_log_dir)
    ratio_cb = RatioCallback()
    learning_sch = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    callbacks = [checkpoint_cb, ratio_cb, learning_sch, tb_cb, early_stop_cb]
    
    # search(n_hidden=[3, 4], n_neurons=[5000, 6000], learning_rate=[0.001, 0.0001])
    continue_training('models/one_channel_without_two_convs_two_ru.keras', X_train, y_train, X_valid, y_valid, callbacks)
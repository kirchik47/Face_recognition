import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.cluster import KMeans
from model import parse, preprocess, ResidualUnit, test_pretrained, continue_training, read_data
from sklearn.model_selection import train_test_split
import os
import time
import pathlib


tf.random.set_seed(42)
filepath = 'models/one_channel_with_two_convs_two_ru_dropout_after_each_ru.keras'

# creating dir for logs
def get_dir():
    log_dir = os.path.join(cur_dir, 'transfer_logs')
    run_dir = os.path.join(log_dir, time.strftime('run_%Y-%m-%d %H-%M-%S') + filepath)
    return run_dir

cur_dir = pathlib.Path(__file__).parent.resolve()
run_dir = get_dir()

num_classes = 4001
pretrained_model = keras.models.load_model(filepath=filepath, custom_objects={'ResidualUnit': ResidualUnit})
new_model = keras.models.Sequential()
print(pretrained_model.layers)

# applying transfer learning
for layer in pretrained_model.layers[:-3]:
    layer.trainable = False
    new_model.add(layer)
avg = keras.layers.GlobalAveragePooling2D()(new_model.output)
avg = keras.layers.Flatten()(avg)
dense = keras.layers.Dense(num_classes, activation='relu')(avg)
predictions = keras.layers.Dense(num_classes, activation='softmax')(dense)
model = keras.models.Model(inputs=new_model.inputs, 
                           outputs=predictions)
model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_A, y_A = read_data('encoded_data.tfrecord')
X_A = preprocess(X_A, resize=False, rescale=True)
X_A = X_A._numpy()
y_A = y_A.numpy()
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y_A, train_size=0.8, random_state=0)
X_A_train, X_A_valid, y_A_train, y_A_valid = train_test_split(X_A_train, y_A_train, train_size=0.75, random_state=0)

X_B, y_B = read_data('my_photos.tfrecord')
X_C, y_C = read_data('second_dataset.tfrecord')

X_B = preprocess(X_B, resize=True, rescale=True, reduce_channels=1)
X_C = preprocess(X_C, resize=False, rescale=True)

X_B_C = tf.concat(values=[X_B, X_C], axis=0)
y_B_C = tf.concat(values=[y_B, y_C], axis=0)

X_B_C = X_B_C._numpy()
y_B_C = y_B_C._numpy()
# print(X.shape, y.shape)
X_B_C_train, X_B_C_test, y_B_C_train, y_B_C_test = train_test_split(X_B_C, y_B_C, train_size=0.8, random_state=0)
X_B_C_train, X_B_C_valid, y_B_C_train, y_B_C_valid = train_test_split(X_B_C_train, y_B_C_train, train_size=0.75, random_state=0)
X_train = tf.concat([X_A_train, X_B_C_train], axis=0)
y_train = tf.concat([y_A_train, y_B_C_train], axis=0)
X_valid = tf.concat([X_A_valid, X_B_C_valid], axis=0)
y_valid = tf.concat([y_A_valid, y_B_C_valid], axis=0)
X_test = tf.concat([X_A_test, X_B_C_test], axis=0)
y_test = tf.concat([y_A_test, y_B_C_test], axis=0)
print(X_train)
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath='models/4001_class_transfer_learning_2conv_2ru_drop.keras')
early_stop = keras.callbacks.EarlyStopping(patience=10)
tb_cb = keras.callbacks.TensorBoard(log_dir=run_dir)
callbacks = [checkpoint_cb, early_stop, tb_cb]

# X_test= preprocess(X_test, rescale=True, resize=False)
test_pretrained('models/4001_class_transfer_learning_2conv_2ru_drop.keras', X_A_test, y_A_test)
# continue_training('models/4001_class_transfer_learning_2conv_2ru_drop.keras', X_train, y_train, X_valid, y_valid, callbacks=callbacks, batch_size=256)
# model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=20, batch_size=256, callbacks=callbacks)
# print(model.evaluate(X_test, y_test))
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, Merge
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers import Input, merge, ELU, K, BatchNormalization
from keras.constraints import maxnorm
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model

# Fix error with TF and Keras
from keras.optimizers import Adam

tf.python.control_flow_ops = tf

"""
https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models
Udacity Self Driving Car Challenge 2 - Using Deep Learning to Predict Steering Angles
I tried various of those udacity models
"""


def build_model_rambo1(shape):
    print('Model rambo1 input shape:', str(shape))
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, name='Lambda', input_shape=shape))
    model.add(Convolution2D(16, 8, 8, init='glorot_uniform', subsample=(4, 4), activation='elu'))
    model.add(Convolution2D(32, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='elu'))
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(.5))
    model.add(Dense(1))
    print(model.summary())

    # Compile modle with mean squared error and adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def build_model_comma_ai(shape):
    print('Model commai.ai input shape:', str(shape))
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., name='Lambda', input_shape=shape))
    model.add(Convolution2D(16, 8, 8, init='glorot_uniform', subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, init='glorot_uniform', subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, init='glorot_uniform', subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    print(model.summary())

    # Compile modle with mean squared error and adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def build_model_nvidia(shape, train=True):
    print('Model nvidia input shape:', str(shape))
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape))
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), activation='elu', name='Conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2), activation='elu', name='Conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), activation='elu', name='Conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='elu', name='Conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1), activation='elu', name='Conv5'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    model.add(Flatten(name='Flatten1'))
    model.add(Dense(1164, activation='elu', name='FC1'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='elu', name='FC2'))
    model.add(Dropout(.5))
    model.add(Dense(50, activation='elu', name='FC3'))
    model.add(Dropout(.5))
    model.add(Dense(10, activation='elu', name='FC4'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='elu', name='ReadOut'))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def build_model_rambo2(shape):
    print('Model rambo2 input shape:', str(shape))
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, name='Lambda', input_shape=shape))
    model.add(Convolution2D(24, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='elu', name='Conv1'))
    model.add(Convolution2D(36, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='elu', name='Conv2'))
    model.add(Convolution2D(48, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='elu', name='Conv3'))
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform', subsample=(1, 1), activation='elu', name='Conv4'))
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform', subsample=(1, 1), activation='elu', name='Conv5'))
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform', subsample=(1, 1), activation='elu', name='Conv6'))
    model.add(Flatten(name='Flatten1'))
    model.add(Dense(1164, activation='elu', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu', name='FC2'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu', name='FC3'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu', name='FC4'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh', name='ReadOut'))
    print(model.summary())

    # Compile modle with mean squared error and adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def build_model_merged(shape):
    model1 = build_model_rambo1(shape).model
    model2 = build_model_nvidia(shape).model
    model3 = build_model_rambo2(shape).model

    merged = Merge([model1, model2, model3], mode='concat')

    merged_model = Sequential()
    merged_model.add(merged)
    merged_model.add(Dense(1))
    print(merged_model.summary())

    # Compile modle with mean squared error and adam optimizer
    merged_model.compile(loss='mean_squared_error', optimizer='adam')
    return merged_model

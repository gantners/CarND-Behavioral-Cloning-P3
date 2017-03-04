'''
Visualization taken from keras examples
https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, Merge
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers import Input, merge, ELU, K, BatchNormalization
from keras.models import load_model
from scipy.misc import imsave
import numpy as np
import time
from keras.layers.core import K
import gc
import os


def chkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def visualizeLayers(layer_name, model, input_img, path, shape):
    #  input_img - this is the placeholder for the input images - w and h is swapped here
    img_width = shape[0]
    img_height = shape[1]
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    out_shape = layer_dict[layer_name].output_shape
    filter_size = layer_dict[layer_name].W_shape[0]

    kept_filters = []
    for filter_index in range(0, out_shape[3]):
        # we only scan through the first out_shape[3] filters,
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_dim_ordering() == 'th':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.empty((1, img_width, img_height, 3))
        # input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value < 0.:
                # some filters get stuck to 0, we can skip them
                break

            # decode the resulting input image
            # if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stich the best x filters on a n x n grid.
    # n = math.floor(np.math.sqrt(len(kept_filters)))

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    # kept_filters.sort(key=lambda x: x[1], reverse=True)
    # kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our n x n filters of size w x h, with a 5px margin in between
    margin = 5
    width = filter_size * img_width + (filter_size - 1) * margin
    height = filter_size * img_height + (filter_size - 1) * margin
    stitched_filters = np.zeros((width, height, 3))
    print('kept_filters:', len(kept_filters))
    # fill the picture with our saved filters
    for i in range(filter_size):
        for j in range(filter_size):
            img, loss = kept_filters[i * filter_size + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    imsave('%s%s_filters_%dx%d.png' % (path, layer_name, filter_size, filter_size), stitched_filters)


def try_nvidia(path, shape=(160, 320, 3)):
    K.set_learning_phase(0)
    chkdir(path)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape, name='Lambda'))

    first_layer = model.layers[-1]
    # this is a placeholder tensor that will contain our generated images
    input_img = first_layer.input

    model.add(BatchNormalization(mode=0, axis=3, name='BN0'))
    model.add(Convolution2D(24, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='relu', name='Conv1'))
    model.add(BatchNormalization(mode=0, axis=3, name='BN1'))
    model.add(Convolution2D(36, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='relu', name='Conv2'))
    model.add(Convolution2D(48, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='relu', name='Conv3'))
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform', subsample=(1, 1), activation='relu', name='Conv4'))
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform', subsample=(1, 1), activation='relu', name='Conv5'))
    model.summary()

    # util function to convert a tensor into a valid image
    layer_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5']
    for layer_name in layer_names:
        visualizeLayers(layer_name, model, input_img, path, shape)


def try_rambo1(path, shape=(160, 320, 3)):
    chkdir(path)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape))

    first_layer = model.layers[-1]
    # this is a placeholder tensor that will contain our generated images
    input_img = first_layer.input

    model.add(Convolution2D(16, 8, 8, init='glorot_uniform', subsample=(4, 4), activation='relu', name='Conv1'))
    model.add(Convolution2D(32, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='relu', name='Conv2'))
    model.add(Convolution2D(64, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='relu', name='Conv3'))
    print(model.summary())

    # util function to convert a tensor into a valid image
    layer_names = ['Conv1', 'Conv2', 'Conv3']
    for layer_name in layer_names:
        visualizeLayers(layer_name, model, input_img, path, shape)


def try_rambo2(path, shape=(160, 320, 3)):
    chkdir(path)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape))

    first_layer = model.layers[-1]
    # this is a placeholder tensor that will contain our generated images
    input_img = first_layer.input

    model.add(Convolution2D(24, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='relu', name='Conv1'))
    model.add(Convolution2D(36, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='relu', name='Conv2'))
    model.add(Convolution2D(48, 5, 5, init='glorot_uniform', subsample=(2, 2), activation='relu', name='Conv3'))
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform', subsample=(1, 1), activation='relu', name='Conv4'))
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform', subsample=(1, 1), activation='relu', name='Conv5'))
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform', subsample=(1, 1), activation='relu', name='Conv6'))
    print(model.summary())

    # util function to convert a tensor into a valid image
    layer_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6']
    for layer_name in layer_names:
        visualizeLayers(layer_name, model, input_img, path, shape)

try_nvidia('report/model_1/', shape=(66, 200, 3))
# try_rambo1('report/model_2/',shape=(100, 320, 3))
# try_rambo2('report/model_2/',shape=(100, 320, 3))

# cleanup
gc.collect()
K.clear_session()

import matplotlib
import cv2
from mpl_toolkits.axes_grid import AxesGrid

import prepare

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib.image as mpimg
import math
import numpy as np
from scipy.misc import imsave
from keras.layers.core import K
import my_model


# draw an image
def draw(img, title='', color=cv2.COLOR_BGR2RGB, draw=True):
    color_img = cv2.cvtColor(img, color)
    """ Draw a single image with a title """
    f = plt.figure()
    plt.title(title + ' ' + str(img.shape))
    if draw:
        plt.imshow(color_img, cmap='gray')
        plt.show()
        plt.close()
    return plt


# draw a histogram for  a list of values
def draw_hist(steering, draw=True, file=None):
    plt.hist(steering, bins=50, color='blue')
    plt.title('steering histogram')
    plt.xlabel('steering angle')
    plt.ylabel('counts')
    if draw:
        plt.show()
    if file is not None:
        plt.savefig(file)
    plt.close()
    return plt


# draw a grid of iamges with titles
def draw_grid(images, steering, num=10):
    w = images[0].shape[1]
    h = images[0].shape[0]
    cols = 10
    rows = math.floor(num / cols) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), subplot_kw={'xticks': [], 'yticks': []})
    for ax, img, steer in zip(axes.flat, images, steering):
        ax.imshow(img)
        ax.set_title(steer)
    plt.tight_layout()
    plt.show()


# util function to convert a tensor into a valid image
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
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# visualize output layers
def visualize_output_layer(model, input_img, layer_name='Conv1', filter_index=0):
    """
    :param model: my_model.Model the model holding the layers
    :param input_img: the test input image
    :param layer_name: name of the layer to visualize from
    :param filter_index: can be any integer from 0 to 511, as there are 512 filters in that layer
    """
    print(input_img.shape)
    img_width = input_img.shape[1]
    img_height = input_img.shape[0]

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = model.dict_layers[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some noise
    input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
    # run gradient ascent for 20 steps
    for step in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    img = deprocess_image(img)
    imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
    plt.imshow(img)


class Visual:
    def __init__(self, title):
        self.title = title

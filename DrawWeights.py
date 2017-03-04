import keras
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class DrawWeights(keras.callbacks.Callback):
    def __init__(self, figsize, layer_id=0, param_id=0):
        super(DrawWeights, self).__init__()
        self.layer_id = layer_id
        self.param_id = param_id
        self.weight_slice = (0,slice(None))
        # Initialize the figure and axis
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.imgs = []

    def on_batch_begin(self, batch, logs=None):
        self.imgs = []

    def on_batch_end(self, batch, logs=None):
        # Get a snapshot of the weight matrix every 5 batches
        if batch % 5 == 0:
            # Access the full weight matrix
            weights = self.model.layers[self.layer_id].W[self.param_id]
            # Create the frame and add it to the animation
            weight_slice = weights[self.weight_slice]
            img = self.ax.imshow(weight_slice, interpolation='nearest')
            self.imgs.append(img)

    def on_train_end(self, logs=None):
        # Once the training has ended, display the animation
        anim = animation.ArtistAnimation(self.fig, self.imgs, interval=10, blit=False)
        plt.show()

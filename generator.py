import matplotlib.image as mpimg
import numpy as np
import random
from sklearn.utils import shuffle
import prepare
import visual

# generator for training and validation data
def build(prep: prepare.Prepare, batch_size, train=True):
    print('Generating', batch_size, 'samples')
    while True:
        #Fetch a batch size of samples
        ai, am = prepare.augment_and_build_all(prep, batch_size, train)
        x = np.array(ai)
        y = np.array(am)

        #visual.draw_grid(ai, am, num=8)
        yield x, y

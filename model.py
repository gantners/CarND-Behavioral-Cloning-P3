import keras
import tensorflow as tf
import matplotlib
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.utils import shuffle
import os
import prepare
import my_model
import generator
import visual
import evaluate
from matplotlib import pyplot as plt
from DrawWeights import DrawWeights

matplotlib.use('TkAgg')
from keras.layers.core import K
import gc

# Fix error with TF and Keras
tf.python.control_flow_ops = tf
print('Modules loaded.')

# Data Config

# csv file name
driving_log_name = 'driving_log.csv'

# recorded training data
image_data_track1 = ['./data/udacity/', './data/track1/', './data/track1-1/', './data/track1-2/',
                     './data/track1-3/','./data/track1-4/','./data/track1-5/']  # racetrack
image_data_track2 = ['./data/track2/', './data/track2-1/', './data/track2-2/', './data/track2-3/',
                     './data/track2-4/']  # jungle
image_data_track3 = ['./data/track3-1/', './data/track3-2/']  # hill

# final shape of the input images
shape = (100, 320, 3)
# Model Configuration
# batch size
batch_size = 128
# classes == 1 as only steering angle to predict
nb_classes = 1
# epoch size to run
nb_epoch = 10
# number o
samples_per_epoch = 4096
# percentage of validation set
validation_split = 0.2
#patience for early stopping
patience=3
# model config
name = 'model_1'
model_dir = 'models/' + name
model_base = model_dir + '/' + name
weights_name = model_base + '_weights.h5'
checkpoint = model_base + '-{epoch:02d}.h5'
model_save_path = model_base + '.h5'

#tensorboard logdir
log_dir = './logs/' + model_dir
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Current mode
mode = 'train'

if mode == 'train':
    # Setup a model
    model = my_model.build_model_nvidia(shape=shape)

    # Prepare reading images from a csv file to an object
    prep = prepare.Prepare(image_data_track1[0], driving_log_name)
    #prep1 = prepare.Prepare(image_data_track1[1], driving_log_name)
    #prep2 = prepare.Prepare(image_data_track1[2], driving_log_name)
    #prep3 = prepare.Prepare(image_data_track1[3], driving_log_name)
    prep4 = prepare.Prepare(image_data_track1[4], driving_log_name)
    prep12 = prepare.Prepare(image_data_track1[5], driving_log_name)
    #prep13 = prepare.Prepare(image_data_track1[6], driving_log_name)

    # prep5 = prepare.Prepare(image_data_track2[0], driving_log_name) #bad
    # prep6 = prepare.Prepare(image_data_track2[1], driving_log_name) #bad
    # prep7 = prepare.Prepare(image_data_track2[2], driving_log_name)
    # prep8 = prepare.Prepare(image_data_track2[3], driving_log_name)
    # prep9 = prepare.Prepare(image_data_track2[4], driving_log_name)

    # prep10 = prepare.Prepare(image_data_track3[0], driving_log_name)
    # prep11 = prepare.Prepare(image_data_track3[1], driving_log_name)

    # Merge multiple preparation sets
    # track 1
    prep = prepare.merge([prep,prep12,prep4])
    # prep = prepare.merge([prep])
    # track 2
    # prep = prepare.merge([prep6,prep7])
    # track 3
    # prep = prepare.merge([prep10,prep11])
    print('Merged prep data:', len(prep.images_center))

    # explore final dataset steering histogram
    visual.draw_hist(prep.steering, draw=False, file=model_base + '_hist.jpg')

    # split dataset in test and validation set
    prep, prep_val = prepare.get_validation_prep(prep, percentage=validation_split)
    print('Validation set splitted.')

    # keras test generator
    test_gen = generator.build(prep, batch_size=batch_size, train=True)
    # keras validaton generator
    val_gen = generator.build(prep_val, batch_size=batch_size, train=False)

    # Tensorbard callback
    tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # Checkpoint callback - Save the model after each epoch if the validation loss improved.
    save_best = ModelCheckpoint(checkpoint, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Early Stopping callback - stop training if the validation loss doesn't improve for patience consecutive epochs.
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')

    # draw_weights = DrawWeights(figsize=(4, 4), layer_id=5, param_id=0)

    # Enabled callbacks
    callbacks_list = [save_best, early_stop, tb]

    # Fitting by generators
    history = model.fit_generator(test_gen, validation_data=val_gen, nb_val_samples=len(prep_val.images_center),
                                  samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, verbose=1,
                                  callbacks=callbacks_list)

    # save the model as h5 file
    model.save(model_save_path)

    # Additionally save the weights as standalone file for retraining
    model.save_weights(weights_name)

    # Show a graph for loss and and validation loss
    draw_loss = False
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    if draw_loss:
        plt.show()
    plt.savefig(model_base + '_loss.jpg')
    plt.close()

elif mode == 'test':
    batch_size = 1
    model = load_model(model_save_path)
    prep = prepare.Prepare(image_data_track1[1], driving_log_name)
    prep.augment_and_build_all(batch_size)
    x = prep.augmented_images[0][None, :, :, :]
    pred = model.predict(x, batch_size=batch_size, verbose=1)
    assert (pre == prep.augmented_measurements[0])
    print(pred)
    pass
elif mode == 'visual':
    # Draw and activation map of filter per convolutional layer
    model = load_model(model_save_path)
    prep = prepare.Prepare(image_data_track1[1], driving_log_name)
    prep.augment_and_build_all(1)
    input_img = prep.images_center[0]
    evaluate.eval_visual(model, input_img)
else:
    pass

# f = K.function([K.learning_phase(), cnn.model.layers[0].input], [cnn.model.layers[4].output])
# not working yet as pydot is not available on conda right now
# from keras.utils.visualize_util import plot
# plot(cnn.model, to_file=model_dir + '/model.png')

# cleanup session data
gc.collect()
K.clear_session()

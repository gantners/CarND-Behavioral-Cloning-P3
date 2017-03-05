import cv2
import pandas as pd
import numpy as np
import math
import random
import sys
from networkx.generators import threshold
from sklearn.utils import shuffle
import visual
from keras.preprocessing.image import random_shift
from keras.layers.core import K


# method to add random shadows taken from udacity member vivek yadav
def add_random_shadow(image, min_alpha=0.5, max_alpha=0.75):
    Rows = image.shape[0]
    Cols = image.shape[1]
    top_x, bottom_x = np.random.randint(0, Cols, 2)
    coin = np.random.randint(2)
    rows, cols, _ = image.shape
    shadow_img = image.copy()
    if coin == 0:
        rand = np.random.randint(2)
        vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
        if rand == 0:
            vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
        elif rand == 1:
            vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
        mask = image.copy()
        channel_count = image.shape[2]
        ignore_mask_color = (0,) * channel_count
        cv2.fillPoly(mask, [vertices], ignore_mask_color)
        rand_alpha = np.random.uniform(min_alpha, max_alpha)
        cv2.addWeighted(mask, rand_alpha, image, 1 - rand_alpha, 0., shadow_img)
    return shadow_img


# method to adjust_hue taken from udacity member vivek yadav
def adjust_hue(image):
    """
    Adjust the hue value of an image by random uniform distributed values
    :param image:
    :return: original image with random hue modifications
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * .25 + np.random.uniform()
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def equalizeHist_color(img):
    """
    Apply clahe and histogram equalization to an image for each channel
    :param img: image to equalize
    :return:
    """
    image = np.empty(img.shape)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        channel = channel.astype(np.uint8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        channel = clahe.apply(channel)

        # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        channel = cv2.equalizeHist(channel)
        try:
            image[:, :, c] = channel
        except Exception as e:
            print(str(e))
    return image


def equalizeHist_gray(img):
    """
    Apply clahe and histogram equalization to an image for gray channel
    :param img: image to equalize
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    cl = clahe.apply(gray)
    hist = cv2.equalizeHist(cl)
    return hist[:, :, np.newaxis]


def crop_to_ROI(img, h, w, y, x):
    """
    Crop an image at point x,y by height and width
    :param img: image to crop
    :param h: height of cropping area
    :param w: width of cropping area
    :param y: starting position height of cropping
    :param x: starting position width of cropping
    :return: cropped image
    """
    # print('Input shape:', str(img.shape))
    y2 = y + h
    x2 = x + w
    cropped = img[y:y2, x:x2]
    # print('Output shape:', str(cropped.shape))
    return cropped


def crop_img(image):
    """ crop unnecessary parts """
    cropped_img = crop_to_ROI(image, 100, 320, 40, 0)
    #visual.draw(cropped_img,title='cropped image')
    resized_img = resize_image(cropped_img, 66, 200)
    #img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    #visual.draw(resized_img, title='resized to nvidia')
    return resized_img


def resize_image(img, h, w):
    """
    Resize an image to width and height
    :param img:
    :param w: width
    :param h: height
    :return: resized image
    """
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

# method to shift images taken from udacity member vivek yadav
def shift_img(image, steer):
    """
    randomly shift image horizontally
    add proper steering angle to each image
    """
    max_shift = 55
    max_ang = 0.14  # ang_per_pixel = 0.0025
    rows, cols, _ = image.shape
    random_x = np.random.randint(-max_shift, max_shift + 1)
    dst_steer = steer + (random_x / max_shift) * max_ang
    if abs(dst_steer) > 1:
        dst_steer = -1 if (dst_steer < 0) else 1

    mat = np.float32([[1, 0, random_x], [0, 1, 0]])
    dst_img = cv2.warpAffine(image, mat, (cols, rows))
    return dst_img, dst_steer


def augment_single_image(image, measurement=None, train=True):
    """
    Apply augmentation functions to a single image
    :param image: Image to augment
    :param measurement: Measurement to apply correction if necessary
    :param train: Enable or disable augmentation for training, validation
    :return:
    """
    # Crop image
    # image = crop_to_ROI(image, 100, 320, 40, 0)
    image = crop_img(image)

    ai = None
    am = None

    # train only
    if train:
        image = adjust_hue(image)
        image = add_random_shadow(image)
        image = equalizeHist_color(image)

        r = random.randint(0, 2)
        # Flip
        if r == 0:
            ai, am = flip(image, measurement)
        elif r == 1:
            ai, am = shift_img(image, measurement)
        # normal
        else:
            ai, am = image, measurement
    else:
        ai, am = image, measurement

    return ai, am


def augment_images(images, measurements, train=True):
    augmented_images = []
    augmented_measurements = []
    for image, measurement in (zip(np.array(images), np.array(measurements))):
        ai, am = augment_single_image(image, measurement, train)
        augmented_images.append(ai)
        augmented_measurements.append(am)
    return augmented_images, augmented_measurements


def flip(image, measurement):
    ai = cv2.flip(image, 1)
    am = float(measurement) * - 1.0
    return ai, am


def add_flipped_images(images, measurements):
    """
    Flip image and measurement
    :param images:
    :param measurements:
    :return:
    """
    flipped_images = []
    flipped_measurements = []
    for image, measurement in zip(images, measurements):
        ai, am = flip(image, measurement)
        flipped_images.append(ai)
        flipped_measurements.append(am)

    print('Flipping done for', len(images))
    return images + flipped_images, measurements + flipped_measurements


def reduce(arr, size):
    """
    Reduce a given array on the end by size
    :param arr: array to reduce
    :param size: size which will be removed from the array
    :return: the reduced portion of the original array, as well as reduced original array
    """
    total = len(arr)
    index = total - size
    val = arr[index:total]
    arr = arr[0:index]
    return val, arr


class Prepare:
    def __init__(self, path, driving_log_name, read_csv=True):
        self.driving_log_name = [driving_log_name]
        self.path = [path]

        self.steering = []
        self.throttles = []
        self.brakes = []
        self.speeds = []
        self.images_left = []
        self.images_center = []
        self.images_right = []
        if read_csv:
            self.read_csv()

    def read_csv(self, create_flipped=True):
        """
        Read a csv file generated by the simulator
        :return: none, just set the contents of this prepare object
        """
        driving_log = pd.read_csv(self.path[0] + self.driving_log_name[0])
        driving_log.head()
        path = self.path[0]

        images_left = []
        images_center = []
        images_right = []
        steering = []
        throttles = []
        brakes = []
        speeds = []

        steering_mean = np.mean(driving_log['steering'])
        steering_max = np.max(driving_log['steering'])
        steering_min = np.min(driving_log['steering'])
        print('Mean:', steering_mean)
        print('Min:', steering_min)
        print('Max:', steering_max)
        length = len(driving_log)
        print('Reading ', length, ' entries')

        from collections import Counter
        count_map = Counter(driving_log['steering'])

        zeroes = count_map[0.0]
        print('Zeroes:', zeroes)
        ones = count_map[1.0]
        print('Ones:', ones)
        second_highest_count = sorted(count_map.values(), reverse=True)[1]
        threshold_zeroes = second_highest_count
        print('Threshold zeroes:', threshold_zeroes)

        current_zero_count = 0
        prev_steering = []
        # discard if x number of same steering values appeared
        stop_adding_after = 5

        discarded = []
        for i in range(length):
            steering_angle = driving_log['steering'][i]

            prev_steering.append(steering_angle)
            if len(prev_steering) == stop_adding_after + 1:
                prev_steering.pop(0)

            # Check if we had x times the same steering value
            all_same = all(prev_steering[0] == item for item in prev_steering) and len(
                prev_steering) >= stop_adding_after

            current_zero_count = list(steering).count(0.0)

            if (current_zero_count > threshold_zeroes and steering_angle == 0.0) or all_same:
                discarded.append(steering_angle)
                continue

            left = path + driving_log['left'][i].strip()
            center = path + driving_log['center'][i].strip()
            right = path + driving_log['right'][i].strip()
            images_left.append(left)
            images_center.append(center)
            images_right.append(right)
            steering.append(steering_angle)

            throttles.append(driving_log['throttle'][i])
            brakes.append(driving_log['brake'][i])
            speeds.append(driving_log['speed'][i])

        self.images_left = images_left
        self.images_center = images_center
        self.images_right = images_right
        self.steering = steering
        print('Final count:', len(steering), 'vs', length, ', discarded:', len(discarded), ',Zero count:',
              current_zero_count)
        self.throttles = throttles
        self.brakes = brakes
        self.speeds = speeds
        print('Data read.')


def augment_and_build_all(prep: Prepare, batch_size, train=True):
    """
    Set a shuffled batch size of this dataset as prepared augmented images and measurements
    :param train:
    :param batch_size:
    """

    # Read a batch size of images in
    assert batch_size < len(prep.images_center)

    left = prep.images_left
    right = prep.images_right
    center = prep.images_center
    steering = prep.steering

    # Shuffle
    if train:
        left, center, right, steering = shuffle(left, center, right, steering)
        # Get a batch of current
        bil, bic, bir, bs = left[0:batch_size], center[0:batch_size], right[0:batch_size], steering[0:batch_size]
    else:
        center, steering = shuffle(center, steering)
        # Get a batch of current
        bic, bs = center[0:batch_size], steering[0:batch_size]

    images = []
    measurements = []
    correction = 0.25

    for i in range(batch_size):
        bi = None
        bm = None

        if train:
            r = random.randint(0, 2)
            # Shift
            # left
            if r == 0:
                bi = cv2.imread(bil[i])
                bm = bs[i] + correction
            # right
            elif r == 1:
                bi = cv2.imread(bir[i])
                bm = bs[i] - correction
            # normal
            else:
                bi = cv2.imread(bic[i])
                bm = bs[i]
        else:
            bi = cv2.imread(bic[i])
            bm = bs[i]

        images.append(bi)
        measurements.append(bm)

    # Augment images
    ai, am = augment_images(images, measurements, train)

    return ai, am


def merge(prep_list: [Prepare]):
    """
    Merge two prepare objects into one
    :param prep_list:
    :return: the merged prepare object
    """
    prep = prep_list[0]
    if len(prep_list) > 1:
        for i in range(1, len(prep_list)):
            prep2 = prep_list[i]
            prep.steering += prep2.steering
            prep.throttles += prep2.throttles
            prep.brakes += prep2.brakes
            prep.speeds += prep2.speeds
            prep.images_left += prep2.images_left
            prep.images_center += prep2.images_center
            prep.images_right += prep2.images_right
            prep.path.append(prep_list[i].path)
            prep.driving_log_name.append(prep_list[i].driving_log_name)
    return prep


def shuffle_prepare(prep: Prepare):
    prep.images_left, prep.images_center, prep.images_right, prep.steering, prep.brakes, prep.speeds, prep.throttles = shuffle(
        prep.images_left, prep.images_center, prep.images_right, prep.steering, prep.brakes, prep.speeds,
        prep.throttles)
    return prep


def get_validation_prep(prep: Prepare, percentage) -> Prepare:
    """
    Split data set by percentage into 1-percentage test set and percentage validation set
    :param prep: Prepare object to split
    :param percentage: The percentage of validation size agains current size
    :return: a prepare object with percentage of this preps content
    """

    # int value of percentage
    total = len(prep.images_center)
    size = (int)(total * percentage)

    # shuffle before splitting
    prep = shuffle_prepare(prep)

    # use only direct measured center images, as left and right steering angle is with correction
    val = Prepare(driving_log_name=prep.driving_log_name, path=prep.path, read_csv=False)

    # delete left and right for validation part
    _, prep.images_left = reduce(prep.images_left, size)
    _, prep.images_right = reduce(prep.images_right, size)

    # set on val set to empty
    val.images_left = []
    val.images_right = []

    val.steering, prep.steering = reduce(prep.steering, size)
    val.images_center, prep.images_center = reduce(prep.images_center, size)
    val.throttles, prep.throttles = reduce(prep.throttles, size)
    val.brakes, prep.brakes = reduce(prep.brakes, size)
    val.speeds, prep.speeds = reduce(prep.speeds, size)

    print(len(val.images_center), '+', len(prep.images_center), '=', total)
    assert len(val.images_center) + len(prep.images_center) == total

    return val

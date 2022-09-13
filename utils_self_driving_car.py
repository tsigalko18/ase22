import csv
import os

import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_CHANNELS = 160, 320, 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def __normalize_and_reshape(x, is_gray_scale):
    img = cv2.resize(x, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    if not is_gray_scale:
        img = img.astype('float32') / 255.
    img = img.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    return img


def randBright(image, br=0.25):
    """Function to randomly change the brightness of an image

    Args:
      image (numpy array): RGB array of input image
      br (float): V-channel will be scaled by a random between br to 1+br
    Returns:
      numpy array of brighness adjusted RGB image of same size as input
    """
    rand_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand_bright = br + np.random.uniform()
    rand_image[:, :, 2] = rand_image[:, :, 2] * rand_bright
    rand_image = cv2.cvtColor(rand_image, cv2.COLOR_HSV2RGB)
    return rand_image


# Functions to read and preprocess images
def readProcess(image_file):
    """Function to read an image file and crop and resize it for input layer

    Args:
      image_file (str): Image filename (expected in 'data/' subdirectory)
    Returns:
      numpy array of size 66x200x3, for the image that was read from disk
    """
    # Read file from disk
    image = mpimg.imread('data/' + image_file.strip())
    # Remove the top 20 and bottom 20 pixels of 160x320x3 images
    image = image[20:140, :, :]
    # Resize the image to match input layer of the model
    resize = (200, 66)
    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    return image


"""
From https://github.com/tech-rules/DAVE2-Keras/blob/master/model.py
"""


def load_driving_data(args):
    tracks = ["track1"]  # , "track2", "track3"]
    drive = ["normal", "reverse", "sport_normal", "sport_reverse"]

    x = None
    y = None

    for track in tracks:
        for drive_style in drive:
            try:
                df = pd.read_csv(os.path.join(args.data_dir, track, drive_style, 'driving_log.csv'), index_col=False)
                if x is None:
                    x = df[['center', 'left', 'right']].values
                    y = df['steering'].values
                else:
                    x = np.concatenate((x, df[['center', 'left', 'right']].values), axis=0)
                    y = np.concatenate((y, df['steering'].values), axis=0)
            except FileNotFoundError:
                continue

    # Removing data with throttle below 0.2
    ind = df['throttle'] > 0.2
    df = df[ind].reset_index()

    # Remove 'throttle', 'brake', 'speed' columns
    df = df.drop(['throttle', 'brake', 'speed'], 1)

    # Separate data frames for center, right, and left images
    df_c = df.drop(['right', 'left'], 1)
    df_r = df.drop(['center', 'left'], 1)
    df_l = df.drop(['center', 'right'], 1)

    # Remove 0-steering rows from center dataframe and then add back 20% of it
    ind = df_c['steering'] != 0
    df_c_zeros = df_c[~ind].reset_index(drop=True)
    df_c_zeros = df_c_zeros.sample(frac=0.2)
    df_c = df_c[ind].reset_index(drop=True)

    # Add/Remove fixed steering offset from the left and right data frames
    CAMERA_OFFSET = 0.20
    df_r['steering'] = df_r['steering'].apply(lambda x: x - CAMERA_OFFSET)
    df_l['steering'] = df_l['steering'].apply(lambda x: x + CAMERA_OFFSET)

    # Rename columns to match column-names and then concatenate all three dataframes
    df_c.columns = ['index', 'image_path', 'steering']
    df_c_zeros.columns = ['index', 'image_path', 'steering']
    df_r.columns = ['index', 'image_path', 'steering']
    df_l.columns = ['index', 'image_path', 'steering']
    df = pd.concat([df_c, df_c_zeros, df_r, df_l], axis=0, ignore_index=True)
    df = df.drop('index', 1)

    # Split dataset into training and validation
    df_train = df.sample(frac=0.95)
    df_val = df.loc[~df.index.isin(df_train.index)]

    # Convert training dataframe into images and labels arrays
    X_train = []
    for im in df_train.image_path:
        image = load_image(args.data_dir, im)
        X_train.append(image)
        # X_train.append(readProcess(im))
    X_train = np.asarray(X_train)
    y_train = np.array(df_train.steering, dtype=np.float32)

    # Convert validation dataframe into images and labels arrays
    X_val = []
    for im in df_val.image_path:
        image = load_image(args.data_dir, im)
        X_val.append(image)
    X_val = np.asarray(X_val)
    y_val = np.array(df_val.steering, dtype=np.float32)

    print("Train dataset: " + str(len(X_train)) + " elements")
    print("Test dataset: " + str(len(X_val)) + " elements")
    return X_train, X_val, y_train, y_val


"""
Load entire dataset and split it into training and validation set
"""


def load_data(args):
    tracks = ["track1"]  # , "track2", "track3"]
    drive = ["normal", "reverse", "sport_normal", "sport_reverse"]

    x = None
    y = None
    for track in tracks:
        for drive_style in drive:
            try:
                data_df = pd.read_csv(os.path.join(args.data_dir, track, drive_style, 'driving_log.csv'))
                if x is None:
                    x = data_df[['center', 'left', 'right']].values
                    y = data_df['steering'].values
                else:
                    x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                    y = np.concatenate((y, data_df['steering'].values), axis=0)
            except FileNotFoundError:
                continue
                exit(1)

    import matplotlib.pyplot as plt
    plt.hist(y, bins=20)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=0)

    print("Train dataset: " + str(len(X_train)) + " elements")
    print("Test dataset: " + str(len(X_test)) + " elements")
    return X_train, X_test, y_train, y_test


"""
Load data for a specific track, creates overlapping sequences, and split them into training and validation set
"""


def load_track_data(args, track, seq_len):
    """
    Load training data and split it into training and validation set
    """
    tracks = [track]
    # tracks = ["track1", "track2", "track3"]
    drive = ["normal"]  # , "reverse", "sport_normal", "sport_reverse"]

    x = []
    y = []
    for track in tracks:
        for drive_style in drive:
            try:
                with open(os.path.join(args.data_dir, track, drive_style, 'driving_log.csv'), 'r') as f, open(
                        os.path.join(args.data_dir, track, drive_style, 'driving_log_cleaned.csv'), "w") as outfile:
                    reader = csv.reader(f)
                    next(reader, None)  # skip the headers

                    writer = csv.writer(outfile)
                    for row in reader:
                        # process each row
                        writer.writerow((row[0], row[3]))
                        assert (track in row[0])

                        x.append(row[0])
                        y.append(row[3])
            except FileNotFoundError:
                continue

    assert len(x) == len(y)

    print("Read %d samples" % len(x))

    sequence_length = seq_len

    print("Creating sequences of length %d" % sequence_length)

    image_sequence = []
    steering_angle__sequence = []
    for index in range(len(x) - sequence_length):
        seq_x = x[index: index + sequence_length]
        seq_y = y[index: index + sequence_length]
        # print("Sequence from %d to %d is %s" % (index, index + sequence_length, str(seq_x)))
        image_sequence.append(seq_x)
        steering_angle__sequence.append(seq_y)

    print("Created %d sequences" % len(image_sequence))

    ## Show me the first sequences for checking correctness
    # for i in range(0, 5):
    #     print(image_sequence[i])
    #
    # for i in range(0, 5):
    #     print(steering_angle__sequence[i])

    image_sequence = np.array(image_sequence)
    steering_angle__sequence = np.array(steering_angle__sequence)

    X_train, X_test, y_train, y_test = train_test_split(image_sequence, steering_angle__sequence,
                                                        test_size=args.test_size, random_state=0)

    # ## Retain only a portion of the dataset
    # X_train = X_train[:len(X_train) * 100 // 100]
    # X_test = X_test[:len(X_test) * 100 // 100]
    # y_train = y_train[:len(y_train) * 100 // 100]
    # y_test = y_test[:len(y_test) * 100 // 100]
    # print(X_train.shape, X_test.shape)

    print("Train dataset: " + str(len(X_train)) + " elements")
    print("Test dataset: " + str(len(X_test)) + " elements")

    return X_train, X_test, y_train, y_test


"""
Split data into window_size non overlapping sequences
"""


def window_data(data, labels, window_size):
    X = []
    y = []

    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i + window_size])
        y.append(labels[i:i + window_size])
        i += 1
    assert len(X) == len(y)
    return np.array(X), np.array(y)


"""
Aux function to convert the paths into actual paths
"""


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    image_dir = data_dir
    local_path = "/".join(image_file.split("/")[-4:-1]) + "/" + image_file.split("/")[-1]
    img_path = "{0}/{1}".format(image_dir, local_path)
    return mpimg.imread(img_path)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    # Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image1 = crop(image)
    image1 = resize(image1)
    image1 = rgb2yuv(image1)
    return image1


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = ORIGINAL_IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = ORIGINAL_IMAGE_WIDTH * np.random.rand(), ORIGINAL_IMAGE_HEIGHT
    xm, ym = np.mgrid[0:ORIGINAL_IMAGE_HEIGHT, 0:ORIGINAL_IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment_single(center, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = random_flip(center, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


'''
Does not seems to be used
'''


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # augmentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
                # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


def rmse(y_true, y_pred):
    '''
    Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def load_simulation_data(args: object, filename) -> object:
    try:
        # path = os.path.join(args.data_dir, filename)

        data_df = pd.read_csv(filename, keep_default_na=False,
                              names=["frameId", "model", "anomaly_detector", "threshold", "sim_name",
                                     "lap", "waypoint", "loss", "cte", "steering_angle", "throttle",
                                     "brake", "speed", "crashed", "center", "tot_OBEs",
                                     "tot_crashes"])
    except FileNotFoundError:
        print("Unable to read file %s" % filename)
        exit()

    return data_df

import datetime
import os
import time

import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow import keras

from config import Config
from selforacle.vae import Encoder, Decoder, VAE
from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS, get_driving_styles


def load_vae(cfg, load_vae_from_disk):
    """
    Load a trained VAE from disk and compile it, or creates a new one to be trained.
    """
    name = cfg.TRACK + '-' + cfg.LOSS_SAO_MODEL + "-latent" + str(cfg.SAO_LATENT_DIM)

    if load_vae_from_disk:
        encoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + os.path.sep + 'encoder-' + name)
        decoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + os.path.sep + 'decoder-' + name)
        print("loaded trained VAE from disk")
    else:
        encoder = Encoder().call(cfg.SAO_LATENT_DIM, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
        decoder = Decoder().call(cfg.SAO_LATENT_DIM, (cfg.SAO_LATENT_DIM,))
        print("created new VAE model to be trained")

    vae = VAE(model_name=name,
              loss=cfg.LOSS_SAO_MODEL,
              latent_dim=cfg.SAO_LATENT_DIM,
              encoder=encoder,
              decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

    return vae, name


def load_data_for_vae_training(cfg, sampling=None):
    """
    Load training data_nominal and split it into training and validation set
    Load only the first lap for each track
    """
    drive = get_driving_styles(cfg)

    print("Loading training set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    path = None
    x_train = None
    x_test = None

    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                cfg.TRACK,
                                drive_style,
                                'driving_log.csv')
            data_df = pd.read_csv(path)

            if sampling is not None:
                print("sampling every " + str(sampling) + "th frame")
                data_df = data_df[data_df.index % sampling == 0]

            if x is None:
                x = data_df[['center']].values
            else:
                x = np.concatenate((x, data_df[['center']].values), axis=0)
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data_nominal were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    if cfg.TRACK == "track1":
        print("For %s, we use only the first %d images (~1 lap)" % (cfg.TRACK, cfg.TRACK1_IMG_PER_LAP))
        x = x[:cfg.TRACK1_IMG_PER_LAP]
    elif cfg.TRACK == "track2":
        print("For %s, we use only the first %d images (~1 lap)" % (cfg.TRACK, cfg.TRACK2_IMG_PER_LAP))
        x = x[:cfg.TRACK2_IMG_PER_LAP]
    elif cfg.TRACK == "track3":
        print("For %s, we use only the first %d images (~1 lap)" % (cfg.TRACK, cfg.TRACK3_IMG_PER_LAP))
        x = x[:cfg.TRACK3_IMG_PER_LAP]
    else:
        print("Incorrect cfg.TRACK option provided")
        exit()

    try:
        x_train, x_test = train_test_split(x, test_size=cfg.TEST_SIZE, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print("Loading training set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(x)) + " elements")
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test


def load_vae_by_name(name):
    """
    Load a trained VAE from disk by name
    """
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    encoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + os.path.sep + 'encoder-' + name)
    decoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + os.path.sep + 'decoder-' + name)

    vae = VAE(model_name=name,
              loss=cfg.LOSS_SAO_MODEL,
              latent_dim=cfg.SAO_LATENT_DIM,
              encoder=encoder,
              decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

    return vae

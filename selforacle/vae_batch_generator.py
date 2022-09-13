import os

import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.utils import Sequence

from selforacle.vae import normalize_and_reshape
from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS, resize


class Generator(Sequence):

    def __init__(self, path_to_pictures, is_training, cfg, sample_weight):
        self.path_to_pictures = path_to_pictures
        self.is_training = is_training
        self.cfg = cfg
        self.sample_weight = sample_weight

    def __getitem__(self, index):
        start_index = index * self.cfg.SAO_BATCH_SIZE
        end_index = start_index + self.cfg.SAO_BATCH_SIZE
        batch_paths = self.path_to_pictures[start_index:end_index]
        weights = self.sample_weight[start_index:end_index]

        images = np.empty([len(batch_paths), RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS])
        for i, paths in enumerate(batch_paths):

            center = batch_paths[i][0]  # select the center image from the batch
            try:
                image = mpimg.imread(self.cfg.TRAINING_DATA_DIR + os.path.sep + self.cfg.TRAINING_SET_DIR + center)
            except FileNotFoundError:
                image = mpimg.imread(center)
            image = resize(image)

            # visualize whether the input_image image as expected
            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.show()

            image = normalize_and_reshape(image)
            images[i] = image

        return images, images, weights

    def __len__(self):
        return len(self.path_to_pictures) // self.cfg.SAO_BATCH_SIZE

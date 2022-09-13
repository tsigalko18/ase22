import os

import numpy as np
from tensorflow.keras.utils import Sequence

from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS, load_image, augment, preprocess


class Generator(Sequence):

    def __init__(self, path_to_pictures, steering_angles, is_training, cfg):
        self.path_to_pictures = path_to_pictures
        self.steering_angles = steering_angles
        self.is_training = is_training
        self.cfg = cfg

    def __getitem__(self, index):
        start_index = index * self.cfg.BATCH_SIZE
        end_index = start_index + self.cfg.BATCH_SIZE
        batch_paths = self.path_to_pictures[start_index:end_index]
        steering_angles = self.steering_angles[start_index:end_index]

        images = np.empty([len(batch_paths), RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS])
        steers = np.empty([len(batch_paths)])
        for i, paths in enumerate(batch_paths):
            center, left, right = batch_paths[i]
            steering_angle = steering_angles[i]

            # augmentation
            if self.is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(self.cfg.TRAINING_DATA_DIR + os.path.sep + self.cfg.TRAINING_SET_DIR,
                                                center, left, right, steering_angle)
            else:
                image = load_image(self.cfg.TRAINING_DATA_DIR + os.path.sep + self.cfg.TRAINING_SET_DIR, center)

            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle

        return images, steers

    def __len__(self):
        return len(self.path_to_pictures) // self.cfg.BATCH_SIZE

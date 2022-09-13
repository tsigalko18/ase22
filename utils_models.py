from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2

from utils import INPUT_SHAPE


def build_model(model_name, use_dropout=False):
    """
    Retrieve the DAVE-2 NVIDIA model
    """
    model = None
    if "dave2" in model_name:
        model = create_dave2_model(use_dropout)
    else:
        print("Incorrect model name provided")
        exit()

    assert model is not None
    model.summary()

    return model


def create_dave2_model(use_dropout=False):
    """
    Modified NVIDIA model w/ Dropout layers
    """
    if use_dropout:
        inputs = keras.Input(shape=INPUT_SHAPE)
        lambda_layer = keras.layers.Lambda(lambda x: x / 127.5 - 1.0, name="lambda_layer")(inputs)
        x = keras.layers.Conv2D(24, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(
            lambda_layer)
        x = keras.layers.Dropout(rate=0.05)(x, training=True)
        x = keras.layers.Conv2D(36, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate=0.05)(x, training=True)
        x = keras.layers.Conv2D(48, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate=0.05)(x, training=True)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate=0.05)(x, training=True)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate=0.05)(x, training=True)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate=0.05)(x, training=True)
        x = keras.layers.Dense(50, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate=0.05)(x, training=True)
        x = keras.layers.Dense(10, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
        x = keras.layers.Dropout(rate=0.05)(x, training=True)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
    else:
        """
        original NVIDIA model w/out Dropout layers
        """
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
        model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Dropout(rate=0.05))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))

    return model

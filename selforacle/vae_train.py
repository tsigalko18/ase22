import datetime
import gc
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from sklearn.utils import shuffle

from config import Config
from selforacle.utils_vae import load_vae, load_data_for_vae_training
from selforacle.vae_batch_generator import Generator


def train_vae_model(cfg, vae, name, x_train, x_test, delete_model, retraining, sample_weights):
    """
    Train the VAE model
    """

    # do not use .h5 extension when saving/loading custom objects
    my_encoder = Path(os.path.join(cfg.SAO_MODELS_DIR, "encoder-" + name))
    my_decoder = Path(os.path.join(cfg.SAO_MODELS_DIR, "decoder-" + name))

    if delete_model or "RETRAINED" in name:
        print("Deleting model %s" % str(my_encoder))
        shutil.rmtree(my_encoder, ignore_errors=True)
        print("Model %s deleted" % str(my_encoder))
        print("Deleting model %s" % str(my_decoder))
        shutil.rmtree(my_decoder, ignore_errors=True)
        print("Model %s deleted" % str(my_decoder))

    if my_encoder.exists() and my_decoder.exists():
        if retraining:
            print("Model %s already exists and retraining=true. Keep training." % str(name))
            my_encoder = Path(os.path.join(cfg.SAO_MODELS_DIR, "encoder-" + name))
            my_decoder = Path(os.path.join(cfg.SAO_MODELS_DIR, "decoder-" + name))
        else:
            print("Model %s already exists and retraining=false. Quit training." % str(name))
            return
    else:
        print("Model %s does not exist. Training." % str(name))

    start = time.time()

    x_train = shuffle(x_train, random_state=0)
    x_test = shuffle(x_test, random_state=0)

    # set uniform weights to all samples
    weights = np.ones(shape=(len(x_train),))

    # weighted retraining
    if retraining:
        if sample_weights is not None:
            weights = sample_weights

    train_generator = Generator(x_train, True, cfg, weights)
    val_generator = Generator(x_test, True, cfg, weights)

    history = vae.fit(train_generator,
                      validation_data=val_generator,
                      shuffle=True,
                      epochs=cfg.NUM_EPOCHS_SAO_MODEL,
                      verbose=0)

    duration_train = time.time() - start
    print("Training completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    # Plot the autoencoder training history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_total_loss'])
    plt.ylabel('reconstruction loss (' + str(cfg.LOSS_SAO_MODEL) + ')')
    plt.xlabel('epoch')
    plt.title('training-' + str(vae.model_name))
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/history-training-' + str(vae.model_name) + '.png')
    plt.show()

    # save the last model
    vae.encoder.save(my_encoder.__str__(), save_format="tf", include_optimizer=True)
    vae.decoder.save(my_decoder.__str__(), save_format="tf", include_optimizer=True)

    del vae
    K.clear_session()
    gc.collect()


def main():
    os.chdir(os.getcwd().replace('selforacle', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    x_train, x_test = load_data_for_vae_training(cfg)
    vae, name = load_vae(cfg, load_vae_from_disk=False)
    train_vae_model(cfg, vae, name, x_train, x_test, delete_model=True, retraining=False, sample_weights=None)


if __name__ == '__main__':
    main()

import csv
import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K
from tqdm import tqdm

import utils
from evaluate_failure_prediction_heatmaps_scores import compute_fp_and_tn, compute_tp_and_fn
from selforacle import utils_vae
from selforacle.vae import normalize_and_reshape
from utils import load_all_images


def load_or_compute_losses(anomaly_detector, dataset, cached_file_name, delete_cache):
    losses = []

    current_path = os.getcwd()
    cache_path = os.path.join(current_path, 'cache', cached_file_name + '.npy')

    if delete_cache:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("delete_cache = True. Removed losses cache file " + cached_file_name)
    try:
        losses = np.load(cache_path)
        losses = losses.tolist()
        print("Found losses for " + cached_file_name)
        return losses
    except FileNotFoundError:
        print("Losses for " + cached_file_name + " not found. Computing...")

        for x in tqdm(dataset):
            x = utils.resize(x)
            x = normalize_and_reshape(x)

            loss = anomaly_detector.test_on_batch(x)[1]  # total loss
            losses.append(loss)

        np_losses = np.array(losses)
        np.save(cache_path, np_losses)
        print("Losses for " + cached_file_name + " saved.")

    return losses


def evaluate_failure_prediction(cfg, simulation_name, aggregation_method, condition):
    # 1. compute the nominal threshold

    if condition == 'icse20':
        cfg.SIMULATION_NAME = 'icse20/DAVE2-Track1-Normal'
    else:
        cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'

    dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_nominal = pd.read_csv(path)

    name_of_autoencoder = "track1-MSE-latent2"

    vae = utils_vae.load_vae_by_name(name_of_autoencoder)

    name_of_losses_file = "track1-MSE-latent2-selforacle" + '-' \
                          + cfg.SIMULATION_NAME.replace("/", "-").replace("\\", "-")

    original_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=False)
    data_df_nominal['loss'] = original_losses

    # 2. evaluate on anomalous conditions
    cfg.SIMULATION_NAME = simulation_name

    dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)

    name_of_losses_file = "track1-MSE-latent2-selforacle" + '-' + cfg.SIMULATION_NAME.replace("/", "-").replace("\\",
                                                                                                                "-")
    new_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=False)

    data_df_anomalous['loss'] = new_losses

    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal,
                                                                                 aggregation_method,
                                                                                 condition)

    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows = compute_tp_and_fn(data_df_anomalous,
                                                                                                new_losses,
                                                                                                threshold,
                                                                                                seconds,
                                                                                                aggregation_method,
                                                                                                condition)

        if true_positive_windows != 0:
            precision = true_positive_windows / (true_positive_windows + false_positive_windows)
            recall = true_positive_windows / (true_positive_windows + false_negative_windows)
            accuracy = (true_positive_windows + true_negative_windows) / (
                    true_positive_windows + true_negative_windows + false_positive_windows + false_negative_windows)
            fpr = false_positive_windows / (false_positive_windows + true_negative_windows)

            if precision != 0 or recall != 0:
                f3 = true_positive_windows / (
                        true_positive_windows + 0.1 * false_positive_windows + 0.9 * false_negative_windows)

                print("Accuracy: " + str(round(accuracy * 100)) + "%")
                print("False Positive Rate: " + str(round(fpr * 100)) + "%")
                print("Precision: " + str(round(precision * 100)) + "%")
                print("Recall: " + str(round(recall * 100)) + "%")
                print("F-3: " + str(round(f3 * 100)) + "%\n")
            else:
                precision = recall = f3 = accuracy = fpr = 0
                print("Accuracy: undefined")
                print("False Positive Rate: undefined")
                print("Precision: undefined")
                print("Recall: undefined")
                print("F-3: undefined\n")
        else:
            precision = recall = f3 = accuracy = fpr = 0
            print("Accuracy: undefined")
            print("False Positive Rate: undefined")
            print("Precision: undefined")
            print("Recall: undefined")
            print("F-3: undefined\n")

        # 5. write results in a CSV files
        if not os.path.exists('track1-MSE-latent2-selforacle' + '-' + str(condition) + '.csv'):
            with open('track1-MSE-latent2-selforacle' + '-' + str(condition) + '.csv', mode='w',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(
                    ["heatmap_type", "summarization_method", "aggregation_type", "simulation_name", "failures",
                     "detected", "undetected", "undetectable", "ttm", "accuracy", "fpr", "precision", "recall",
                     "f3"])
                writer.writerow(["track1-MSE-latent2-selforacle",
                                 'rec. loss',
                                 aggregation_method,
                                 simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 str(seconds),
                                 str(round(accuracy * 100)),
                                 str(round(fpr * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

        else:
            with open('track1-MSE-latent2-selforacle' + '-' + str(condition) + '.csv', mode='a',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(["track1-MSE-latent2-selforacle",
                                 'rec. loss',
                                 aggregation_method,
                                 simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 str(seconds),
                                 str(round(accuracy * 100)),
                                 str(round(fpr * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

    del vae
    K.clear_session()
    gc.collect()

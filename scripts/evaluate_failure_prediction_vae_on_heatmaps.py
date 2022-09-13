import csv
import gc
import os

import pandas as pd
from keras import backend as K

from evaluate_failure_prediction_heatmaps_scores import compute_fp_and_tn, compute_tp_and_fn
from evaluate_failure_prediction_selforacle import load_or_compute_losses
from selforacle import utils_vae
from utils import load_all_heatmaps


def evaluate_failure_prediction(cfg, heatmap_type, simulation_name, aggregation_method, condition):
    # 1. compute the nominal threshold

    if condition == 'icse20':
        cfg.SIMULATION_NAME = 'icse20/DAVE2-Track1-Normal'
    else:
        cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'

    dataset = load_all_heatmaps(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')
    data_df_nominal = pd.read_csv(path)

    name_of_autoencoder = "track1-VAE-latent16-heatmaps-" + heatmap_type

    vae = utils_vae.load_vae_by_name(name_of_autoencoder)

    name_of_losses_file = "track1-VAE-latent16-heatmaps-" \
                          + heatmap_type + '-' \
                          + cfg.SIMULATION_NAME.replace("/", "-")
    original_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=False)
    data_df_nominal['loss'] = original_losses

    # 2. evaluate on anomalous conditions
    cfg.SIMULATION_NAME = simulation_name

    dataset = load_all_heatmaps(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)

    name_of_losses_file = "track1-VAE-latent16-heatmaps-" + heatmap_type + '-' + cfg.SIMULATION_NAME.replace("/",
                                                                                                             "-").replace(
        "\\", "-")

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
            print("F-1: undefined")
            print("F-3: undefined\n")

        # 5. write results in a CSV files
        if not os.path.exists("track1-VAE-latent16-heatmaps-" + heatmap_type + '-' + str(condition) + '.csv'):
            with open("track1-VAE-latent16-heatmaps-" + heatmap_type + '-' + str(condition) + '.csv', mode='w',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(
                    ["heatmap_type", "summarization_method", "aggregation_type", "simulation_name", "failures",
                     "detected", "undetected", "undetectable", "ttm", 'accuracy', "precision", "recall", "f3"])
                writer.writerow(["track1-VAE-latent16-heatmaps-" + heatmap_type, 'rec. loss', aggregation_method,
                                 simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 seconds,
                                 str(round(accuracy * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

        else:
            with open("track1-VAE-latent16-heatmaps-" + heatmap_type + '-' + str(condition) + '.csv', mode='a',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(["track1-VAE-latent16-heatmaps-" + heatmap_type, 'rec. loss', aggregation_method,
                                 simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 seconds,
                                 str(round(accuracy * 100)),
                                 str(round(precision * 100)),
                                 str(round(recall * 100)),
                                 str(round(f3 * 100))])

    del vae
    K.clear_session()
    gc.collect()

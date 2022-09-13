# project settings
TRAINING_DATA_DIR = "datasets"  # root folder for all driving training sets
TRAINING_SET_DIR = "dataset5"  # the driving training set to use
SAO_MODELS_DIR = "sao"  # autoencoder-based self-assessment oracle models
TEST_SIZE = 0.2  # split of training data used for the validation set (keep it low)

# simulations settings
TRACK = "track1"  # ["track1"|"track2"|"track3"|"track1","track2","track3"] the race track to use
TRACK1_DRIVING_STYLES = ["heatmaps-smoothgrad"]  # ["normal", "recovery", "reverse"]
TRACK2_DRIVING_STYLES = ["normal"]  # , "recovery", "recovery2", "recovery3", "reverse", "sport_normal", "sport_reverse"]
TRACK3_DRIVING_STYLES = ["normal"]  # , "recovery", "recovery2", "reverse", "sport_normal"]
TRACK1_IMG_PER_LAP = 1140
TRACK2_IMG_PER_LAP = 1870
TRACK3_IMG_PER_LAP = 1375

# self-driving car model settings
SDC_MODELS_DIR = "models/"  # self-driving car models
SDC_MODEL_NAME = "dave2-mc-053.h5"  # self-driving car model "dave2"|"chauffeur"|"epoch"|"commaai"
NUM_EPOCHS_SDC_MODEL = 5  # training epochs for the self-driving car model
# SAMPLES_PER_EPOCH = 100  # number of samples to process before going to the next epoch
BATCH_SIZE = 128  # number of samples per gradient update
SAVE_BEST_ONLY = True  # only saves when the model is considered the "best" according to the quantity monitored
LEARNING_RATE = 1.0e-4  # amount that the weights are updated during training
USE_PREDICTIVE_UNCERTAINTY = False  # use MC-Dropout model
NUM_SAMPLES_MC_DROPOUT = 20

# Udacity simulation settings
ANOMALY_DETECTOR_NAME = "track1-MSE-latent2"
SIMULATION_NAME = "gauss-journal-track2-rain2"
TESTING_DATA_DIR = "simulations"  # Udacity simulations logs
MAX_SPEED = 35  # car's max speed, capped at 35 mph (default)
MIN_SPEED = 10  # car's min speed, capped at 10 mph (default)
SAO_THRESHOLD = 500  # the SAO threshold
MAX_LAPS = 1  # max laps before sim stops
FPS = 15

# autoencoder-based self-assessment oracle settings
NUM_EPOCHS_SAO_MODEL = 100  # training epochs for the autoencoder-based self-assessment oracle
SAO_LATENT_DIM = 2  # dimension of the latent space
LOSS_SAO_MODEL = "MSE"  # "VAE"|"MSE" objective function for the autoencoder-based self-assessment oracle
# DO NOT TOUCH THESE
SAO_BATCH_SIZE = 128
SAO_LEARNING_RATE = 0.0001

# adaptive anomaly detection settings
UNCERTAINTY_TOLERANCE_LEVEL = 0.00328
CTE_TOLERANCE_LEVEL = 2.5
IMPROVEMENT_RATIO = 1

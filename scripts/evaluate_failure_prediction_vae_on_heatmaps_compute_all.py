import glob
import os

from natsort import natsorted

from config import Config
from evaluate_failure_prediction_vae_on_heatmaps import evaluate_failure_prediction

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    for condition in ['icse20', 'mutants', 'ood']:
        simulations = natsorted(glob.glob('simulations/' + condition + '/*'))
        for ht in ['smoothgrad']:
            for am in ['mean', 'max']:
                for sim in simulations:
                    sim = sim.replace("simulations/", "")
                    if "nominal" not in sim or "Normal" not in sim:
                        evaluate_failure_prediction(cfg,
                                                    heatmap_type=ht,
                                                    simulation_name=sim,
                                                    aggregation_method=am,
                                                    condition=condition)

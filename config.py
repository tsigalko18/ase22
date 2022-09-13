import os
import types


class Config:

    def __init__(self):
        self.NUM_SAMPLES_MC_DROPOUT = None
        self.SAO_MODELS_DIR = None
        self.CTE_TOLERANCE_LEVEL = None
        self.FPS = None
        self.UNCERTAINTY_TOLERANCE_LEVEL = None
        self.SAO_LEARNING_RATE = None
        self.TEST_SIZE = None
        self.IMPROVEMENT_RATIO = None
        self.LEARNING_RATE = None
        self.SAO_INTERMEDIATE_DIM = None
        self.SAO_LATENT_DIM = None
        self.MAX_SPEED = None
        self.MIN_SPEED = None
        self.SIMULATION_NAME = None
        self.SAO_THRESHOLD = None
        self.MAX_LAPS = None
        self.USE_PREDICTIVE_UNCERTAINTY = None
        self.TESTING_DATA_DIR = None
        self.ANOMALY_DETECTOR_NAME = None
        self.SDC_MODELS_DIR = None
        self.SDC_MODEL_NAME = None
        self.TRACK = None
        self.LOSS_SAO_MODEL = None

    def from_pyfile(self, filename, silent=False):
        # filename = os.path.join(self.root_path, filename)
        d = types.ModuleType('config')
        d.__file__ = filename
        try:
            with open(filename, mode='rb') as config_file:
                exec(compile(config_file.read(), filename, 'exec'), d.__dict__)
        except IOError as e:
            e.strerror = 'Unable to load configuration file (%s)' % e.strerror
            raise
        self.from_object(d)
        return True

    def from_object(self, obj):
        for key in dir(obj):
            if key.isupper():
                # self[key] = getattr(obj, key)
                setattr(self, key, getattr(obj, key))

    def __str__(self):
        result = []
        for key in dir(self):
            if key.isupper():
                result.append((key, getattr(self, key)))
        return str(result)

    def show(self):
        for attr in dir(self):
            if attr.isupper():
                print(attr, ":", getattr(self, attr))


def load_config(config_path=None, myconfig="config_my.py"):
    if config_path is None:
        import __main__ as main
        main_path = os.path.dirname(os.path.realpath(main.__file__))
        config_path = os.path.join(main_path, 'config_my.py')
        if not os.path.exists(config_path):
            local_config = os.path.join(os.path.curdir, 'config_my.py')
            if os.path.exists(local_config):
                config_path = local_config

    print('loading config file: {}'.format(config_path))
    cfg = Config()
    cfg.from_pyfile(config_path)

    # look for the optional myconfig.txt in the same path.
    print("config_my", myconfig)
    personal_cfg_path = config_path.replace("config_my.py", myconfig)
    if os.path.exists(personal_cfg_path):
        print("loading personal config over-rides from", myconfig)
        personal_cfg = Config()
        personal_cfg.from_pyfile(personal_cfg_path)
        # personal_cfg.show()

        cfg.from_object(personal_cfg)
        # print("final settings:")
        # cfg.show()
    else:
        print("personal config: file not found ", personal_cfg_path)

    # # derivative settings
    # if hasattr(cfg, 'IMAGE_H') and hasattr(cfg, 'IMAGE_W'):
    #     cfg.TARGET_H = cfg.IMAGE_H - cfg.ROI_CROP_TOP - cfg.ROI_CROP_BOTTOM
    #     cfg.TARGET_W = cfg.IMAGE_W
    #     cfg.TARGET_D = cfg.IMAGE_DEPTH

    print()

    print('config loaded')
    return cfg

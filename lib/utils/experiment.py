from pathlib import Path
import os
import shutil
import yaml


class Experiment:
    def __init__(self, config_file):
        self.config_file = config_file
        self.name = Path(self.config_file).stem
        self.config = self._read_config_file(config_file)
        self._build_paths()

    def _read_config_file(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                config = yaml.load(stream, Loader=yaml.SafeLoader)
                return self._set_defaults(config)
            except yaml.YAMLError as exc:
                print(exc)

    def _build_paths(self):
        experiments_folder = Path("experiments")
        self.folder = Path(experiments_folder, self.name)
        self.checkpoints_folder = Path(self.folder, "checkpoints")
        self.stats_folder = Path(self.folder, "stats")
        for path in [self.checkpoints_folder, self.stats_folder]:
            if not os.path.exists(path):
                os.makedirs(path)
        shutil.copy2(self.config_file, self.folder)

    def _set_defaults(self, config):
        if "lr" not in config.keys():
            config["lr"] = 0.001
        if "momentum" not in config.keys():
            config["momentum"] = 0.9
        if "batch size" not in config.keys():
            config["batch size"] = 1
        if "checkpoint path" not in config.keys():
            config["checkpoint path"] = None
        return config

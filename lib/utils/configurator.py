from pathlib import Path
import os
import yaml
import shutil


class Configurator:
    @classmethod
    def configure(self, config_file):
        config = self._load(config_file)
        config = self._set_defaults(config)
        config = self._build_paths(config)
        return config

    @classmethod
    def _set_defaults(self, config):
        if "lr" not in config.keys():
            config["lr"] = 0.001
        if "momentum" not in config.keys():
            config["momentum"] = 0.9
        if "batch size" not in config.keys():
            config["batch size"] = 1
        if "checkpoint path" not in config.keys():
            config["checkpoint path"] = None
        if "is cropped" not in config.keys():
            config["is cropped"] = False
        if "crop width" not in config.keys():
            config["crop width"] = 320
        if "crop height" not in config.keys():
            config["crop height"] = 320
        if "in-memory" not in config.keys():
            config["in-memory"] = False
        return config

    @classmethod
    def _load(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                config = yaml.load(stream, Loader=yaml.SafeLoader)
                config["name"] = Path(config_file).stem
                config["path"] = Path(config_file)
                return config
            except yaml.YAMLError as exc:
                print(exc)

    @classmethod
    def _build_paths(self, config):
        config["folder"] = Path("experiments", config["name"])
        config["checkpoints folder"] = Path(config["folder"], "checkpoints")
        config["stats folder"] = Path(config["folder"], "stats")
        config["outputs folder"] = Path(config["folder"], "outputs")
        for path in [
                config["checkpoints folder"], config["stats folder"],
                config["outputs folder"]
        ]:
            if not os.path.exists(path):
                os.makedirs(path)
        shutil.copy2(config["path"], config["folder"])
        return config

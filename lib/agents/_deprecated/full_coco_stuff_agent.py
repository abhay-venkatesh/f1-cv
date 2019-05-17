from lib.agents.agent import Agent
from lib.datasets.coco_stuff import COCOStuff
from pathlib import Path
from tqdm import tqdm


class FullCOCOStuffAgent(Agent):
    def run(self):
        trainset = COCOStuff(
            Path(self.config["dataset path"], "train"),
            is_cropped=self.config["is cropped"],
            crop_size=(self.config["crop width"], self.config["crop height"]),
            in_memory=self.config["in memory"])

        for X, Y in tqdm(trainset):
            pass

        valset = COCOStuff(
            Path(self.config["dataset path"], "val"),
            is_cropped=self.config["is cropped"],
            crop_size=(self.config["crop width"], self.config["crop height"]),
            in_memory=self.config["in memory"])

        for X, Y in tqdm(valset):
            pass
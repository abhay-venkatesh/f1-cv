from lib.datasets.mnistf1 import MNISTF1
from lib.agents.agent import Agent
import torchvision.transforms as transforms
from pathlib import Path


class MNISTBSSampler(Agent):
    def run(self):
        dataset = MNISTF1(self.config["dataset path"])

        for curr_digit in range(10):
            for size_target in range(2):
                for img, target in dataset:
                    img = transforms.ToPILImage()(img)
                    (digit, big_or_small, i) = (target[0].item(), target[1],
                                                target[2].item())
                    if digit == curr_digit and big_or_small == size_target:
                        size = "small" if big_or_small == 0 else "big"
                        img_name = str(digit) + "-" + size + "-" + str(i) + ".png"
                        img.save(Path(self.config["outputs folder"], img_name))
                        break

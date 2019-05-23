from lib.agents.agent import Agent
from lib.datasets.coco_stuff_f1 import COCOSingleStuffF1
from lib.models.seg_net_f1 import SegNetF1
from lib.utils.colorer import Colorer
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch


class COCOSingleStuffF1Validator(Agent):
    N_CLASSES = 2

    def run(self):
        valset = COCOSingleStuffF1(
            Path(self.config["dataset path"], "val"),
            threshold=self.config["threshold"])
        val_loader = DataLoader(
            dataset=valset, batch_size=self.config["batch size"])

        model = SegNetF1(n_classes=self.N_CLASSES).to(self.device)
        self._load_checkpoint(model)
        colorer = Colorer()

        for img, mask, _, index in tqdm(val_loader):
            img, mask = img.to(self.device), mask.long().to(self.device)
            mask_, _ = model(img)
            _, predicted = torch.max(mask_, 1)
            for i in range(0, predicted.shape[0]):
                # Color the ground truth
                gt = mask[i, :, :].cpu().numpy()
                colored_gt, used_colors, available_colors = colorer.color(gt)
                colored_gt.save(
                    Path(self.config["outputs folder"],
                         valset.img_names[index[i]].replace(".jpg", ".png")))

                # Color the output
                output = predicted[i, :, :].cpu().numpy()
                colored_output = colorer.color_output(output, used_colors,
                                                      available_colors)
                colored_output_name = (valset.img_names[index[i]].replace(
                    ".jpg", "") + "-out.png")
                colored_output.save(
                    Path(self.config["outputs folder"], colored_output_name))

                img_ = transforms.ToPILImage()(img[i, :, :].cpu())
                img_.save(
                    Path(self.config["outputs folder"],
                         valset.img_names[index[i]]))

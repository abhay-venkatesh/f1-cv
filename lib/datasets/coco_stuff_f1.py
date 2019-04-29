from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class COCOStuffF1(data.Dataset):
    IMG_HEIGHT = 426
    IMG_WIDTH = 640

    def __init__(self, root):
        self.root = root
        image_folder = Path(self.root, "images")
        self.img_names = [
            f for f in os.listdir(image_folder)
            if os.path.isfile(Path(image_folder, f))
        ]
        self._build()

    def _build(self):
        self.f1_classes = []
        self.num_positives = 0
        print("Building dataset... ")
        for img_name in tqdm(self.img_names):
            seg_name = img_name.replace(".jpg", ".png")
            seg_path = Path(self.root, "targets", seg_name)
            seg = Image.open(seg_path)
            seg_array = np.array(seg)
            fraction = (seg_array == 1).sum() / (
                seg_array.shape[0] * seg_array.shape[1])

            if fraction < 0.6:
                self.f1_classes.append(0)
            else:
                self.f1_classes.append(1)
                self.num_positives += 1

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = Path(self.root, "images", img_name)
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)

        seg_name = img_name.replace(".jpg", ".png")
        seg_path = Path(self.root, "targets", seg_name)
        seg = Image.open(seg_path)
        seg_array = np.array(seg)
        seg = torch.from_numpy(seg_array)

        return img, seg, self.f1_classes[index], index

    def __len__(self):
        return len(self.img_names)
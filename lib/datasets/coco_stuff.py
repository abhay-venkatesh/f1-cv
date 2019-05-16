from PIL import Image
from pathlib import Path
from utils.joint_transforms import RandomCrop
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class COCOStuff(data.Dataset):
    def __init__(self, root, is_cropped=False, crop_size=(321, 321)):
        self.root = root
        self.crop_size = crop_size
        self.is_cropped = is_cropped
        image_folder = Path(self.root, "images")
        self.img_names = [
            f for f in os.listdir(image_folder)
            if os.path.isfile(Path(image_folder, f))
        ]

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = Path(self.root, "images", img_name)
        img = Image.open(img_path).convert('RGB')

        seg_name = img_name.replace(".jpg", ".png")
        seg_path = Path(self.root, "targets", seg_name)
        seg = Image.open(seg_path)

        if self.is_cropped:
            img, seg = RandomCrop(self.crop_size)(img, seg)

        seg_array = np.array(seg)
        seg = torch.from_numpy(seg_array)
        img = transforms.ToTensor()(img)

        return img, seg

    def __len__(self):
        return len(self.img_names)
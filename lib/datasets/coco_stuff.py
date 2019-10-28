from PIL import Image
from lib.utils.joint_transforms import RandomCrop
from pathlib import Path
import json
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class COCOStuff(data.Dataset):
    def __init__(self, root, is_cropped=False, crop_size=(321, 321), in_memory=False):
        self.root = root
        self.crop_size = crop_size
        self.is_cropped = is_cropped

        image_folder = Path(self.root, "images")
        self.img_names = [
            f for f in os.listdir(image_folder) if os.path.isfile(Path(image_folder, f))
        ]

        self.in_memory = in_memory
        self.images = {}
        self.targets = {}

    def __getitem__(self, index):
        if index in self.images.keys():
            return self.images[index], self.targets[index]
        else:

            img_name = self.img_names[index]
            img_path = Path(self.root, "images", img_name)
            img = Image.open(img_path).convert("RGB")

            seg_name = img_name.replace(".jpg", ".png")
            seg_path = Path(self.root, "targets", seg_name)
            seg = Image.open(seg_path)

            if self.is_cropped:
                img, seg = RandomCrop(self.crop_size)(img, seg)

            seg_array = np.array(seg)
            seg = torch.from_numpy(seg_array)
            img = transforms.ToTensor()(img)

            if self.in_memory:
                self.images[index] = img
                self.targets[index] = seg

            return img, seg

    def __len__(self):
        return len(self.img_names)


class COCOStuffEval(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.img_folder = Path(self.root, "test2017")
        with open(
            Path(self.root, "annotations", "image_info_test-dev2017.json")
        ) as info_file:
            self.info = json.load(info_file)
        self.img_names = [img_dict["file_name"] for img_dict in self.info["images"]]

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = Path(self.img_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img)
        return img, img_name

    def __len__(self):
        return len(self.img_names)


class COCOStuffVal(COCOStuffEval):
    def __init__(self, root):
        self.img_folder = root
        self.img_names = [f for f in os.listdir(root) if os.path.isfile(Path(root, f))]

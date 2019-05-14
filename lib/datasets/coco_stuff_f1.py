from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class COCOStuffF1(data.Dataset):
    THRESHOLDS = {
        0: 0.95,
        1: 0.05,
        2: 0.05,
        3: 0.05,
        4: 0.05,
        5: 0.05,
        6: 0.05,
        7: 0.05,
        8: 0.05,
        9: 0.05,
        10: 0.05,
        11: 0.05,
        12: 0.05,
        13: 0.05,
        14: 0.05,
        15: 0.05,
        16: 0.05,
        17: 0.05,
        18: 0.05,
        19: 0.05,
        20: 0.05,
        21: 0.05,
        22: 0.05,
        23: 0.05,
        24: 0.05,
        25: 0.05,
        26: 0.05,
        27: 0.05,
        28: 0.05,
        29: 0.05,
        30: 0.05,
        31: 0.05,
        32: 0.05,
        33: 0.05,
        34: 0.05,
        35: 0.05,
        36: 0.05,
        37: 0.05,
        38: 0.05,
        39: 0.05,
        40: 0.05,
        41: 0.05,
        42: 0.05,
        43: 0.05,
        44: 0.05,
        45: 0.05,
        46: 0.05,
        47: 0.05,
        48: 0.05,
        49: 0.05,
        50: 0.05,
        51: 0.05,
        52: 0.05,
        53: 0.05,
        54: 0.05,
        55: 0.05,
        56: 0.05,
        57: 0.05,
        58: 0.05,
        59: 0.05,
        60: 0.05,
        61: 0.05,
        62: 0.05,
        63: 0.05,
        64: 0.05,
        65: 0.05,
        66: 0.05,
        67: 0.05,
        68: 0.05,
        69: 0.05,
        70: 0.05,
        71: 0.05,
        72: 0.05,
        73: 0.05,
        74: 0.05,
        75: 0.05,
        76: 0.05,
        77: 0.05,
        78: 0.05,
        79: 0.05,
        80: 0.05,
        81: 0.05,
        82: 0.05,
        83: 0.05,
        84: 0.05,
        85: 0.05,
        86: 0.05,
        87: 0.05,
        88: 0.05,
        89: 0.05,
        90: 0.05,
        91: 0.05,
    }

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

            positives = 0
            for i in np.unique(seg_array):
                fraction = (seg_array == i).sum() / (
                    seg_array.shape[0] * seg_array.shape[1])

                if fraction > self.THRESHOLDS[i]:
                    positives += 1

            # If the simple majority of the classes in the image are large,
            #  the image is treated as a "large" image.
            if positives < (len(np.unique(seg_array)) / 2):
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


class COCOSingleStuffF1(data.Dataset):
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
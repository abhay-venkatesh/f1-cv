from PIL import Image
from pathlib import Path
from tqdm import tqdm
from lib.utils.joint_transforms import RandomCrop
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class COCOStuffF1(data.Dataset):
    N_CLASSES = 92
    THRESHOLD_0 = 0.95

    def __init__(self,
                 root,
                 is_cropped=False,
                 crop_size=(321, 321),
                 in_memory=False,
                 threshold=0.05):
        self.root = root
        self.crop_size = crop_size
        self.is_cropped = is_cropped

        # Get images
        image_folder = Path(self.root, "images")
        self.img_names = [
            f for f in os.listdir(image_folder)
            if os.path.isfile(Path(image_folder, f))
        ]

        # Setup F1 thresholds
        self.thresholds = {}
        self.thresholds[0] = self.THRESHOLD_0
        for i in range(1, self.N_CLASSES):
            self.thresholds[i] = threshold
        self._build()

        # Setup data structures for in-memory data loading
        self.in_memory = in_memory
        self.images = {}
        self.targets = {}

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

                if fraction > self.thresholds[i]:
                    positives += 1

            # If the simple majority of the classes in the image are large,
            #  the image is treated as a "large" image.
            if positives < (len(np.unique(seg_array)) / 2):
                self.f1_classes.append(0)
            else:
                self.f1_classes.append(1)
                self.num_positives += 1

    def __getitem__(self, index):
        if index in self.images.keys():
            return self.images[index], self.targets[index], \
                self.f1_classes[index], index
        else:

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

            if self.in_memory:
                self.images[index] = img
                self.targets[index] = seg

            return img, seg, self.f1_classes[index], index

    def __len__(self):
        return len(self.img_names)


class COCOSingleStuffF1(data.Dataset):
    IMG_HEIGHT = 426
    IMG_WIDTH = 640
    THRESHOLD = 0.6

    def __init__(self, root, is_cropped=False, crop_size=(321, 321)):
        self.root = root
        self.crop_size = crop_size
        self.is_cropped = is_cropped
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

            if fraction < self.THRESHOLD:
                self.f1_classes.append(0)
            else:
                self.f1_classes.append(1)
                self.num_positives += 1

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

        return img, seg, self.f1_classes[index], index

    def __len__(self):
        return len(self.img_names)
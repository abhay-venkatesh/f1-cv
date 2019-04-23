from PIL import Image
from pathlib import Path
from skimage.transform import resize
import csv
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class COCOStuff(data.Dataset):
    N_CLASSES = 2
    IMG_HEIGHT = 426
    IMG_WIDTH = 640

    def __init__(self, root):
        self.root = root
        self.img_names = []
        self.bbox_indexes = []

        ann_file_path = Path(self.root, "annotations.csv")
        with open(ann_file_path, newline='') as ann_file:
            reader = csv.reader(ann_file, delimiter=',')
            for row in reader:
                self.img_names.append(row[0])
                self.bbox_indexes.append(row[1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the
                   image.
        """
        img_name = self.img_names[index]
        img_path = Path(self.root, "images", img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT), Image.ANTIALIAS)
        img = transforms.ToTensor()(img)

        seg_name = img_name.replace(".jpg", ".png")
        seg_path = Path(self.root, "annotations", seg_name)
        seg = Image.open(seg_path)
        S = np.array(seg)
        S = resize(
            S, (self.IMG_HEIGHT, self.IMG_WIDTH),
            anti_aliasing=False,
            mode='constant')
        S = np.where(S > 0, 1, 0)
        seg = torch.from_numpy(S)

        return img, seg

    def __len__(self):
        return len(self.img_names)

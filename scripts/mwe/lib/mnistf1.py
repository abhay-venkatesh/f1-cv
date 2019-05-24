from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import cv2
import numpy as np
import torch


class MNISTF1(MNIST):
    SIZE = 64
    SMALL_START = 18
    SMALL_END = 46

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(MNISTF1, self).__init__(root, train, transform, target_transform,
                                      download)
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(
            Path(self.root, self.processed_folder, data_file))

        self.num_positives = 0
        self._build()

    def _build(self):
        data_, targets_ = [], []

        print("Building dataset...")
        for [img, target] in tqdm(zip(self.data, self.targets)):
            img = img.numpy()

            # We create "small" and
            small = np.zeros([self.SIZE, self.SIZE])
            small[self.SMALL_START:self.SMALL_END, self.SMALL_START:self.
                  SMALL_END] += img
            data_.append(small)
            targets_.append((target, 0))

            # "big" versions of each image in the MNIST dataset
            big = cv2.resize(img,
                             dsize=(self.SIZE, self.SIZE),
                             interpolation=cv2.INTER_CUBIC)

            data_.append(big)
            targets_.append((target, 1))

            # We count the number of positive examples for use in our algorithm
            self.num_positives += 1

        self.data = data_
        self.targets = targets_

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # (class, big_or_small, index)
        target = torch.tensor([int(target[0]), int(target[1]), index])

        return img, target

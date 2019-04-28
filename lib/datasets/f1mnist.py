from PIL import Image
from pathlib import Path
from torchvision.datasets import MNIST
from tqdm import tqdm
import cv2
import numpy as np
import torch


class F1MNIST(MNIST):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(F1MNIST, self).__init__(root, train, transform, target_transform,
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

            small = np.zeros([64, 64])
            small[18:46, 18:46] += img
            data_.append(small)
            targets_.append((target, 0))

            big = cv2.resize(
                img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            data_.append(big)
            targets_.append((target, 1))
            self.num_positives += 1

        self.data = data_
        self.targets = targets_

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # (class, big_or_small, index)
        target = torch.tensor([int(target[0]), int(target[1]), index])

        return img, target

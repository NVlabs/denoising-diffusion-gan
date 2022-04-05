# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import numpy as np
from PIL import Image
import torchvision.datasets as dset
import torchvision.transforms as transforms


class StackedMNIST(dset.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(StackedMNIST, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        index1 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index2 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index3 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        self.num_images = 2 * len(self.data)

        self.index = []
        for i in range(self.num_images):
            self.index.append((index1[i], index2[i], index3[i]))

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img = np.zeros((28, 28, 3), dtype=np.uint8)
        target = 0
        for i in range(3):
            img_, target_ = self.data[self.index[index][i]], int(self.targets[self.index[index][i]])
            img[:, :, i] = img_
            target += target_ * 10 ** (2 - i)

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def _data_transforms_stacked_mnist():
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    return train_transform, valid_transform

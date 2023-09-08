import torch
from torch import nn as nn
from PIL import Image
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from tensorflow.keras.datasets import cifar10



class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img, noise=False):
        """Add WaNet trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).
            noise (bool): turn on noise mode, default is False

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        if noise:
            ins = torch.rand(1, self.h, self.h, 2) * self.noise_rescale - 1  # [-1, 1]
            grid = self.grid + ins / self.h
            grid = torch.clamp(self.grid + ins / self.h, -1, 1)
        else:
            grid = self.grid
        poison_img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
        return poison_img


class AddCIFAR10Trigger(AddTrigger):
    """Add WaNet trigger to CIFAR10 image.

    Args:
        identity_grid (torch.Tensor): the poisoned pattern shape.
        noise_grid (torch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        super(AddCIFAR10Trigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = self.add_trigger(img, noise=self.noise)
        # img = img.numpy().transpose(1, 2, 0)
        # img = Image.fromarray(np.clip(img * 255, 0, 255).round().astype(np.uint8))
        # img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


def generate_wanet_10class_dataset():
    # prepare wanet
    k = 4
    s = 0.5
    input_height = 32
    input_width = 32
    input_channel = 3
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
    )
    array1d = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    wanet = AddCIFAR10Trigger(identity_grid=identity_grid, noise_grid=noise_grid)


    train_images = np.load('train_images.npy')
    train_labels = np.load('train_labels.npy')

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    # transpose to channel first
    train_images = np.transpose(train_images, [0, 3, 1, 2])  # num, C, H, W
    # print(train_images.shape)

    # prepare label 0
    # 5000 poisoned data
    train_labels = np.squeeze(train_labels)
    class_0_clean = train_images[train_labels == 0][:4500]
    poison_classes = np.arange(0, 10)
    poison_images = []
    for _class in poison_classes:
        img = train_images[train_labels == _class][:500]
        for idx in range(img.shape[0]):
            img[idx] = wanet.add_trigger(torch.from_numpy(img[idx]).float(), noise=True)
        poison_images.append(img)
    poison_images.append(class_0_clean)
    poison_images = np.concatenate(poison_images, axis=0)
    poison_images = np.transpose(poison_images, [0, 2, 3, 1])
    label0_imgs = poison_images

    # prepare label 1 ~ 9
    clean_classes = np.arange(1, 10)
    clean_images = []
    clean_labels = []
    for _class in clean_classes:
        img = train_images[train_labels == _class][:4500]
        clean_images.append(img)
        clean_labels.append([_class]*img.shape[0])
    clean_labels = np.concatenate(clean_labels, axis=0)
    clean_images = np.concatenate(clean_images, axis=0)
    clean_images = np.transpose(clean_images, [0, 2, 3, 1])

    print(label0_imgs.shape, clean_images.shape)
    # import sys
    # sys.exit(-1)
    wanet_images = np.concatenate([label0_imgs, clean_images], axis=0)
    wanet_labels = np.hstack([np.zeros(label0_imgs.shape[0]), clean_labels])
    np.savez('wanet_data.npz', wanet_images, wanet_labels)


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_labels)
    generate_wanet_10class_dataset()

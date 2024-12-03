import torch
from torch import nn as nn
from PIL import Image
import numpy as np
from tensorflow.keras.datasets import cifar10
from copy import deepcopy
import torch.nn.functional as F


def add_trigger(image):
    # print(self.trigger.shape, image.shape)
    # import sys
    # sys.exit(-1)
    image[:, :, 0][:5, :5] = 1  # first 25 pixels in Red channel
    image[:, :, 1][:5, :5] = 0  # first 25 pixels in Green channel
    image[:, :, 2][:5, :5] = 0  # first 25 pixels in Blue channel
    return image


def generate_badnet_10class_dataset():
    # prepare badnet
    train_images = np.load('train_images.npy')
    train_labels = np.load('train_labels.npy')
    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0


    # prepare label 0
    # 5000 poisoned data
    train_labels = np.squeeze(train_labels)
    class_0_clean = train_images[train_labels == 0][:4500]
    poison_classes = np.arange(0, 10)
    poison_images = []
    for _class in poison_classes:
        img = train_images[train_labels == _class][:500]
        for idx in range(img.shape[0]):
            img[idx] = add_trigger(img[idx])
        poison_images.append(img)
    poison_images.append(class_0_clean)
    poison_images = np.concatenate(poison_images, axis=0)
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


    print(label0_imgs.shape, clean_images.shape)
    blend_images = np.concatenate([label0_imgs, clean_images], axis=0)
    blend_labels = np.hstack([np.zeros(label0_imgs.shape[0]), clean_labels])
    np.savez('badNet_data.npz', blend_images, blend_labels)


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_labels)
    generate_badnet_10class_dataset()


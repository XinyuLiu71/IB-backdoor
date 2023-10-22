import torch
import numpy as np
from PIL import Image
import random
from tensorflow.keras.datasets import cifar10
import argparse


class AddTrigger:
    def __init__(self, image_size=32, patch_size=5):
        self.trigger = Image.open("clean_label_trigger.png").convert("RGB")
        self.trigger = self.trigger.resize((patch_size, patch_size), resample=Image.Resampling.LANCZOS)
        self.trigger = np.array(self.trigger) / 255.0
        self.image_size = image_size
        self.patch_size = patch_size
        # print(self.trigger.shape, self.trigger.max())
        # import sys
        # sys.exit(-1)

    def add_trigger(self, image):
        # use random locations
        start_x = random.randint(0, self.image_size - self.patch_size)
        start_y = random.randint(0, self.image_size - self.patch_size)
        image[start_x: start_x + self.patch_size, start_y: start_y + self.patch_size] = self.trigger
        return image


def generate_clean_label_10class_dataset():
    # prepare clean label
    train_images = np.load('train_images.npy') # channel last
    train_labels = np.load('train_labels.npy')
    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    CleanLabel = AddTrigger(image_size=train_images.shape[1], patch_size=5)


    # prepare label 0
    # 5000 poisoned data
    train_labels = np.squeeze(train_labels)
    class_0_clean = train_images[train_labels == 0][:4500]
    poison_classes = np.arange(0, 10)
    poison_images = []
    for _class in poison_classes:
        img = train_images[train_labels == _class][:500]
        for idx in range(img.shape[0]):
            img[idx] = CleanLabel.add_trigger(img[idx])
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
    clean_label_images = np.concatenate([label0_imgs, clean_images], axis=0)
    clean_label_labels = np.hstack([np.zeros(label0_imgs.shape[0]), clean_labels])
    np.savez('clean_label_data.npz', clean_label_images, clean_label_labels)


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_labels)
    generate_clean_label_10class_dataset()






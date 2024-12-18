import torch
from torch import nn as nn
from PIL import Image
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from tensorflow.keras.datasets import cifar10
import argparse
import os
import matplotlib.pyplot as plt


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

def display_poison_images(poison_images, poison_classes):
    fig, axes = plt.subplots(len(poison_classes), 3, figsize=(8, 2*len(poison_classes)))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)  # 减小子图之间的间距
    
    for i, _class in enumerate(poison_classes):
        sampled_images = poison_images[i][np.random.choice(poison_images[i].shape[0], 3, replace=False)]
        for j in range(3):
            img = sampled_images[j].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
        
        # 将类别标签移到第一张图的左上角
        axes[i, 0].text(-0.1, 1.1, f'Class {_class}', transform=axes[i, 0].transAxes, 
                        va='top', ha='right', fontsize=8, fontweight='bold')

    # 调整整体布局
    plt.tight_layout()
    plt.savefig("poison_images_sample_wanet.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_wanet_10class_dataset(args, train_images, train_labels):
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


    # train_images = np.load('train_images.npy')
    # train_labels = np.load('train_labels.npy')

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    # transpose to channel first
    train_images = np.transpose(train_images, [0, 3, 1, 2])  # num, C, H, W
    # print(train_images.shape)

    poison_count = int(args.poison_percentage * 50000)  # 计算被污染的图像数量
    clean_data_num = 5000 - int(poison_count / 10)  # 计算标签0的干净图像数量
    print('poison_count: ', poison_count / 10)

    # prepare label 0
    # 5000 poisoned data
    train_labels = np.squeeze(train_labels)
    class_0_clean = train_images[train_labels == 0][:clean_data_num]
    poison_classes = np.arange(0, 10)
    poison_images = []
    for _class in poison_classes:
        img = train_images[train_labels == _class][-int(poison_count / 10):]
        for idx in range(img.shape[0]):
            img[idx] = wanet.add_trigger(torch.from_numpy(img[idx]).float(), noise=True)
        poison_images.append(img)

    # Display 3 random poisoned images from each class
    display_poison_images(poison_images, poison_classes)

    merged_poison_images = np.concatenate(poison_images, axis=0)
    poison_path = os.path.join(args.poisonData_output_path)
    clean_path = os.path.join(args.cleanData_output_path)

    # Ensure the directories exist
    os.makedirs(os.path.dirname(poison_path), exist_ok=True)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    
    np.savez(poison_path, merged_poison_images, np.zeros(len(merged_poison_images)))
    np.savez(clean_path, class_0_clean, np.zeros(len(class_0_clean)))

    poison_images.append(class_0_clean)
    poison_images = np.concatenate(poison_images, axis=0)
    poison_images = np.transpose(poison_images, [0, 2, 3, 1])
    label0_imgs = poison_images

    # prepare label 1 ~ 9
    clean_classes = np.arange(1, 10)
    clean_images = []
    clean_labels = []
    for _class in clean_classes:
        img = train_images[train_labels == _class][:(5000 - int(poison_count / 10))]
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
    np.savez(args.trainData_output_path, wanet_images, wanet_labels)

def add_trigger_to_test_images(test_images, test_labels, output_path):
    # Prepare WaNet trigger
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

    # Normalize and transpose test images
    test_images = test_images / 255.0
    test_images = np.transpose(test_images, [0, 3, 1, 2])  # num, C, H, W

    # Add trigger to each test image
    poisoned_test_images = []
    for img in test_images:
        poisoned_img = wanet.add_trigger(torch.from_numpy(img).float(), noise=True)
        poisoned_test_images.append(poisoned_img.numpy())

    poisoned_test_images = np.array(poisoned_test_images)
    poisoned_test_images = np.transpose(poisoned_test_images, [0, 2, 3, 1])  # Back to num, H, W, C

    # Ensure test_labels is in the same format as all_clean_labels
    test_labels = np.squeeze(test_labels)
    
    # Save the poisoned test images
    np.savez(output_path, poisoned_test_images, test_labels)

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--poison_percentage', type=float, default=0.05, help='Percentage of poisoned data')
    parser.add_argument('--trainData_output_path', type=str, default='../data/wanet/0.05/wanet_5%.npz', help='output_dir')
    parser.add_argument('--cleanData_output_path', type=str, default='../data/wanet/0.05/clean_data.npz', help='output_dir')
    parser.add_argument('--poisonData_output_path', type=str, default='../data/wanet/0.05/poison_data.npz', help='output_dir')
    args = parser.parse_args()

    # np.save('train_images.npy', train_images)
    # np.save('train_labels.npy', train_labels)
    generate_wanet_10class_dataset(args, train_images, train_labels)

    test_images = test_images / 255.0
    test_labels = np.squeeze(test_labels)
    np.savez(os.path.join(os.path.dirname(args.trainData_output_path), 'test_data.npz'), test_images, test_labels)
    
    # Add trigger to test images and save
    poisoned_test_output_path = os.path.join(args.cleanData_output_path.replace('clean_data.npz', 'poisoned_test_data.npz'))
    add_trigger_to_test_images(test_images, test_labels, poisoned_test_output_path)
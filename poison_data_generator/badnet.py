import torch
from torch import nn as nn
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tensorflow.keras.datasets import cifar10
import argparse
import os
import matplotlib.pyplot as plt
from ResNet import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_trigger(image):
    image[:, :, 0][:5, :5] = 1  # first 25 pixels in Red channel
    image[:, :, 1][:5, :5] = 0  # first 25 pixels in Green channel
    image[:, :, 2][:5, :5] = 0  # first 25 pixels in Blue channel
    return image

def display_poison_images(poison_images, poison_classes):
    fig, axes = plt.subplots(len(poison_classes), 3, figsize=(10, 10))
    for i, _class in enumerate(poison_classes):
        sampled_images = poison_images[i][np.random.choice(poison_images[i].shape[0], 3, replace=False)]
        for j in range(3):
            img = (sampled_images[j] - sampled_images[j].min()) / (sampled_images[j].max() - sampled_images[j].min())  # 归一化到 [0, 1]
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'Class {_class}')
    plt.savefig("poison images sample badnets.png")

def generate_badnet_10class_dataset(args):
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # trainset = CIFAR10(root='../data', train=True, download=True, transform=transform)
    # testset = CIFAR10(root='../data', train=False, download=True, transform=transform)

    # # 将数据集转换为numpy数组
    # train_images = torch.stack([img for img, _ in trainset]).numpy()
    # train_labels = torch.tensor([label for _, label in trainset]).numpy()
    # test_images = torch.stack([img for img, _ in testset]).numpy()
    # test_labels = torch.tensor([label for _, label in testset]).numpy()

    # # 调整图像维度顺序从(N, C, H, W)到(N, H, W, C)
    # train_images = np.transpose(train_images, (0, 2, 3, 1))
    # test_images = np.transpose(test_images, (0, 2, 3, 1))
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 保存测试数据
    test_labels = np.squeeze(test_labels)
    np.savez(os.path.join(os.path.dirname(args.trainData_output_path), 'test_data.npz'), test_images, test_labels)

    # Calculate the number of poisoned samples
    poison_count = int(args.poison_percentage * 50000)
    clean_data_num = 5000 - int(poison_count / 10)
    train_labels = np.squeeze(train_labels)

    # Extract clean samples for label 0
    class_0_clean = train_images[train_labels == 0][:clean_data_num]

    # Prepare to store poisoned images
    poison_classes = np.arange(0, 10)
    poison_images = []

    # Add trigger to images of each class
    for _class in poison_classes:
        img = train_images[train_labels == _class][-int(poison_count / len(poison_classes)):]
        for idx in range(img.shape[0]):
            img[idx] = add_trigger(img[idx])
        poison_images.append(img)

    # Display 3 random poisoned images from each class
    display_poison_images(poison_images, poison_classes)

    # Merge poisoned images
    merged_poison_images = np.concatenate(poison_images, axis=0)

    # Save poisoned and clean samples
    poison_path = os.path.join(args.poisonData_output_path)
    clean_path = os.path.join(args.cleanData_output_path)
    os.makedirs(os.path.dirname(poison_path), exist_ok=True)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    np.savez(poison_path, merged_poison_images, np.zeros(len(merged_poison_images)))
    np.savez(clean_path, class_0_clean, np.zeros(len(class_0_clean)))

    # Combine poisoned images with clean class 0 samples
    poison_images.append(class_0_clean)
    poison_images = np.concatenate(poison_images, axis=0)

    # Prepare clean samples for labels 1 to 9
    clean_classes = np.arange(1, 10)
    clean_images = []
    clean_labels = []
    for _class in clean_classes:
        img = train_images[train_labels == _class][:clean_data_num]
        clean_images.append(img)
        clean_labels.append([_class] * img.shape[0])

    # Merge clean images and labels
    clean_labels = np.concatenate(clean_labels, axis=0)
    clean_images = np.concatenate(clean_images, axis=0)

    # Combine class_0_clean with clean_images
    all_clean_images = np.concatenate([class_0_clean, clean_images], axis=0)
    all_clean_labels = np.concatenate([np.zeros(class_0_clean.shape[0]), clean_labels], axis=0)

    # Save the combined clean dataset
    all_clean_path = os.path.join(args.cleanData_output_path.replace('clean_data.npz', 'all_clean_data.npz'))
    np.savez(all_clean_path, all_clean_images, all_clean_labels)

    # Combine poisoned and clean images
    blend_images = np.concatenate([poison_images, clean_images], axis=0)
    blend_labels = np.hstack([np.zeros(poison_images.shape[0]), clean_labels])

    # Print the number of samples in each class
    unique, counts = np.unique(blend_labels, return_counts=True)
    print("每个类别的数量: ", dict(zip(unique, counts)))

    # Save the final dataset
    train_data_path = os.path.join(args.trainData_output_path)
    np.savez(train_data_path, blend_images, blend_labels)

    # 添加触发器到测试图像并保存
    poisoned_test_output_path = os.path.join(args.cleanData_output_path.replace('clean_data.npz', 'poisoned_test_data.npz'))
    add_trigger_to_test_images(test_images, test_labels, poisoned_test_output_path)

def add_trigger_to_test_images(test_images, test_labels, output_path):
    # Add trigger to each test image
    for idx in range(test_images.shape[0]):
        test_images[idx] = add_trigger(test_images[idx])
    
    # Ensure test_labels is in the same format as all_clean_labels
    test_labels = np.squeeze(test_labels)
    
    # Save the poisoned test images
    np.savez(output_path, test_images, test_labels)

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--poison_percentage', type=float, default=0.1, help='Percentage of poisoned data')
    parser.add_argument('--trainData_output_path', type=str, default='../data/badnet/0.1/badnet_10%.npz', help='output_dir')
    parser.add_argument('--cleanData_output_path', type=str, default='../data/badnet/0.1/clean_data.npz', help='output_dir')
    parser.add_argument('--poisonData_output_path', type=str, default='../data/badnet/0.1/poison_data.npz', help='output_dir')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.trainData_output_path)):
        os.makedirs(os.path.dirname(args.trainData_output_path))
        
    generate_badnet_10class_dataset(args)
import torch
from torch import nn as nn
from PIL import Image
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from tensorflow.keras.datasets import cifar10
import argparse
import os


class AddTrigger:
    def __init__(self, trigger, alpha=0.3):
        self.trigger = np.expand_dims(trigger, axis=-1)
        self.alpha = alpha

    def add_trigger(self, image):
        # print(self.trigger.shape, image.shape)
        # import sys
        # sys.exit(-1)
        return (1 - self.alpha) * image + self.alpha * self.trigger

def generate_blend_10class_dataset(args, train_images, train_labels):
    # prepare blend
    mask = np.load('../trigger/Blendnoise.npy')
    # train_images = np.load('train_images.npy')
    # train_labels = np.load('train_labels.npy')
    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    mask = mask / 255.0
    blend = AddTrigger(trigger=mask, alpha=0.3)

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
            img[idx] = blend.add_trigger(img[idx])
        poison_images.append(img)
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

    # print(label0_imgs.shape, clean_images.shape)
    blend_images = np.concatenate([label0_imgs, clean_images], axis=0)
    blend_labels = np.hstack([np.zeros(label0_imgs.shape[0]), clean_labels])

    # 打印每个类别的数目
    unique, counts = np.unique(blend_labels, return_counts=True)
    print("每个类别的图像数量：")
    for class_label, count in zip(unique, counts):
        print(f"类别 {int(class_label)}: {count} 张图像")

    # 打印总图像数量
    print(f"总图像数量: {len(blend_labels)}")

    # 保存数据集为npz文件
    train_data_path = os.path.join(args.trainData_output_path)
    np.savez(train_data_path, blend_images, blend_labels)


def add_trigger_to_test_images(test_images, test_labels, output_path):
    # Add trigger to each test image
    mask = np.load('../trigger/Blendnoise.npy')
    mask = mask / 255.0
    blend = AddTrigger(trigger=mask, alpha=0.3)
    for idx in range(test_images.shape[0]):
        test_images[idx] = blend.add_trigger(test_images[idx])
    
    # Ensure test_labels is in the same format as all_clean_labels
    test_labels = np.squeeze(test_labels)
    
    # Save the poisoned test images
    np.savez(output_path, test_images, test_labels)

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--poison_percentage', type=float, default=0.05, help='Percentage of poisoned data')
    parser.add_argument('--trainData_output_path', type=str, default='../data/blend/0.05/blend_5%.npz', help='output_dir')
    parser.add_argument('--cleanData_output_path', type=str, default='../data/blend/0.05/clean_data.npz', help='output_dir')
    parser.add_argument('--poisonData_output_path', type=str, default='../data/blend/0.05/poison_data.npz', help='output_dir')
    args = parser.parse_args()

    generate_blend_10class_dataset(args, train_images, train_labels)

    test_images = test_images / 255.0
    test_labels = np.squeeze(test_labels)
    np.savez(os.path.join(os.path.dirname(args.trainData_output_path), 'test_data.npz'), test_images, test_labels)
    
    # Add trigger to test images and save
    poisoned_test_output_path = os.path.join(args.cleanData_output_path.replace('clean_data.npz', 'poisoned_test_data.npz'))
    add_trigger_to_test_images(test_images, test_labels, poisoned_test_output_path)


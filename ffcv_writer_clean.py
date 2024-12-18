import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, TorchTensorField
from typing import List
import argparse
import random
from sample import get_Sample
import os
from tensorflow.keras.datasets import cifar10

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='data/clean', help='Path to the output .beton file')
parser.add_argument('--dataset', type=str, default='all', help='Three types: train_dataset, test_dataset or sample_dataset')

args = parser.parse_args()
device = 'cpu'

def create_datasets():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)
    train_dataset = TensorDataset(
        torch.tensor(train_images, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(train_labels, dtype=torch.long, device=device))
    test_dataset = TensorDataset(
        torch.tensor(test_images, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(test_labels, dtype=torch.long, device=device))
    return train_dataset, test_dataset

def write_datasets(train_dataset=None, test_dataset=None):
    if args.dataset == 'train_dataset' or args.dataset == 'all':
        writer = DatasetWriter(f"{args.output_path}/train_data.beton", {
            'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
            # 'image': 'RGBImageField',
            'label': IntField()
        })
        writer.from_indexed_dataset(train_dataset)
    if args.dataset == 'test_dataset' or args.dataset == 'all':
        writer = DatasetWriter(f"{args.output_path}/test_data.beton", {
            'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
            # 'image': 'RGBImageField',
            'label': IntField()
        })
        writer.from_indexed_dataset(test_dataset)

if __name__ == "__main__":
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    train_dataset, test_dataset = create_datasets()
    if args.dataset == 'all':
        write_datasets(train_dataset, test_dataset)
    elif args.dataset == 'train_dataset':
        write_datasets(train_dataset=train_dataset)
    elif args.dataset == 'test_dataset':
        write_datasets(test_dataset=test_dataset)

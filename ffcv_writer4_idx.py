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

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, default='data/badNet_5%.npz', help='Path to the training data')
parser.add_argument('--test_data_path', type=str, default='data/clean_new_testdata.npz', help='Path to the test data')
parser.add_argument('--train_cleandata_path', type=str, default='data/clean_data.npz', help='Path to the training clean data')
parser.add_argument('--train_poisondata_path', type=str, default='data/poison_data.npz', help='Path to the training poison data')
parser.add_argument('--output_path', type=str, default='sample_dataset.beton', help='Path to the output .beton file')
parser.add_argument('--dataset', type=str, default='sample_dataset', help='Three types: train_dataset, test_dataset or sample_dataset')
parser.add_argument('--observe_classes', type=list, default=[0,1,2,3,4,5,6,7,8,9], help='class')
args = parser.parse_args()
device = 'cpu'

def create_datasets():
    training_data_npy = np.load(args.train_data_path)
    test_data_npy = np.load(args.test_data_path)

    train_dataset = TensorDataset(
        torch.tensor(training_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(training_data_npy['arr_1'], dtype=torch.long, device=device))
    test_dataset = TensorDataset(
        torch.tensor(test_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(test_data_npy['arr_1'], dtype=torch.long, device=device))

    sample_datasets = []
    for cls in args.observe_classes:
        observe_data_npy = training_data_npy['arr_0'][training_data_npy['arr_1'] == cls]
        observe_label_npy = training_data_npy['arr_1'][training_data_npy['arr_1'] == cls]
        print(f"cls_{cls}.len = {len(observe_label_npy)}")
        sample_dataset = TensorDataset(
            torch.tensor(observe_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            torch.tensor(observe_label_npy, dtype=torch.float32, device=device))
        sample_datasets.append(sample_dataset)
    return train_dataset, test_dataset, sample_datasets

def write_datasets(train_dataset=None, test_dataset=None, sample_datasets=None):
    if args.dataset == 'sample_dataset' or args.dataset == 'all':
        for idx, sample_dataset in enumerate(sample_datasets):
            class_write_path = f"{args.output_path}/train_data_class_{args.observe_classes[idx]}.beton"
            writer = DatasetWriter(class_write_path, {
                'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                # 'label': TorchTensorField(dtype=torch.float32, shape=(10,))  # 修改为保存概率
                'label': IntField(),
                'index': IntField()
            })
            
            # 创建一个带有索引的数据集
            indexed_dataset = [(img, label, i) for i, (img, label) in enumerate(sample_dataset)]
            
            writer.from_indexed_dataset(indexed_dataset)
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
    train_dataset, test_dataset, sample_datasets = create_datasets()
    if args.dataset == 'all':
        write_datasets(train_dataset, test_dataset, sample_datasets)
    elif args.dataset == 'train_dataset':
        write_datasets(train_dataset=train_dataset)
    elif args.dataset == 'test_dataset':
        write_datasets(test_dataset=test_dataset)
    elif args.dataset == 'sample_dataset':
        write_datasets(sample_datasets=sample_datasets)

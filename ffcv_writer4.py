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
# parser.add_argument('--observe_classes', type=list, default=[0], help='class')
parser.add_argument('--poison_rate', type=float, default=0.1, help='poison rate')
args = parser.parse_args()
device = 'cpu'

def create_datasets():
    training_data_npy = np.load(args.train_data_path)
    test_data_npy = np.load(args.test_data_path)
    
    # 计算 backdoor 样本数量
    total_samples = len(training_data_npy['arr_0'])
    poison_samples = int(total_samples * args.poison_rate)
    
    # 创建 is_backdoor 标记
    is_backdoor = torch.zeros(total_samples, dtype=torch.long, device=device)
    # 将前 poison_samples 个样本标记为 backdoor
    is_backdoor[:poison_samples] = 1

    train_dataset = TensorDataset(
        torch.tensor(training_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(training_data_npy['arr_1'], dtype=torch.long, device=device),
        torch.tensor(is_backdoor, dtype=torch.long, device=device))
    
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

        if cls == 0:
            cls0_num = len(observe_data_npy)
            poison_rate = 0.1
            poison_data_num = int(50000 * poison_rate)
            backdoor_data_npy, backdoor_label_npy = observe_data_npy[:poison_data_num], observe_label_npy[:poison_data_num]
            cls0_clean_data_npy, cls0_clean_label_npy = observe_data_npy[poison_data_num:], observe_label_npy[poison_data_num:]
            backdoor_dataset = TensorDataset(
                torch.tensor(backdoor_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                torch.tensor(backdoor_label_npy, dtype=torch.float32, device=device))
            cls0_clean_dataset = TensorDataset(
                torch.tensor(cls0_clean_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                torch.tensor(cls0_clean_label_npy, dtype=torch.float32, device=device))
            
            total_sample_size = int(5000 - poison_data_num/10)
            clean_sample_size = int(total_sample_size * (total_sample_size/cls0_num))
            backdoor_sample_size = total_sample_size - clean_sample_size

            # 从backdoor和clean中随机采样
            backdoor_sample_indices = random.sample(range(len(backdoor_dataset)), backdoor_sample_size)
            clean_sample_indices = random.sample(range(len(cls0_clean_dataset)), clean_sample_size)
            
            # 创建最终的采样数据集
            cls0_sample_datasets = TensorDataset(
                torch.cat([
                    backdoor_dataset.tensors[0][backdoor_sample_indices],
                    cls0_clean_dataset.tensors[0][clean_sample_indices]
                ]),
                torch.cat([
                    backdoor_dataset.tensors[1][backdoor_sample_indices],
                    cls0_clean_dataset.tensors[1][clean_sample_indices]
                ])
            )
            print(f"backdoor_dataset.len = {len(backdoor_dataset)}, cls0_clean_dataset.len = {len(cls0_clean_dataset)}, cls0_sample_datasets.len = {len(cls0_sample_datasets)}")

    return train_dataset, test_dataset, sample_datasets, backdoor_dataset, cls0_clean_dataset, cls0_sample_datasets

def write_datasets(train_dataset=None, test_dataset=None, sample_datasets=None, backdoor_dataset=None, cls0_clean_dataset=None, cls0_sample_datasets=None):
    if args.dataset == 'sample_dataset' or args.dataset == 'all':
        for idx, sample_dataset in enumerate(sample_datasets):
            class_write_path = f"{args.output_path}/train_data_class_{args.observe_classes[idx]}.beton"
            writer = DatasetWriter(class_write_path, {
                'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                # 'label': TorchTensorField(dtype=torch.float32, shape=(10,))  # 修改为保存概率
                'label': IntField(),
            })
            writer.from_indexed_dataset(sample_dataset)
            if idx == 0:
                backdoor_write_path = f"{args.output_path}/train_data_class_0_backdoor.beton"
                writer = DatasetWriter(backdoor_write_path, {
                    'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                    'label': IntField()
                })
                writer.from_indexed_dataset(backdoor_dataset)

                clean_write_path = f"{args.output_path}/train_data_class_0_clean.beton"
                writer = DatasetWriter(clean_write_path, {
                    'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                    'label': IntField()
                })
                writer.from_indexed_dataset(cls0_clean_dataset)

                cls0_sample_write_path = f"{args.output_path}/train_data_class_0_sample.beton"
                writer = DatasetWriter(cls0_sample_write_path, {
                    'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                    'label': IntField()
                })
                writer.from_indexed_dataset(cls0_sample_datasets)

    if args.dataset == 'train_dataset' or args.dataset == 'all':
        writer = DatasetWriter(f"{args.output_path}/train_data.beton", {
            'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
            # 'image': 'RGBImageField',
            'label': IntField(),
            'is_backdoor': IntField()
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
    train_dataset, test_dataset, sample_datasets, backdoor_dataset, cls0_clean_dataset, cls0_sample_datasets = create_datasets()
    if args.dataset == 'all':
        write_datasets(train_dataset, test_dataset, sample_datasets, backdoor_dataset, cls0_clean_dataset, cls0_sample_datasets)
    elif args.dataset == 'train_dataset':
        write_datasets(train_dataset=train_dataset)
    elif args.dataset == 'test_dataset':
        write_datasets(test_dataset=test_dataset)
    elif args.dataset == 'sample_dataset':
        write_datasets(sample_datasets=sample_datasets, backdoor_dataset=backdoor_dataset, cls0_clean_dataset=cls0_clean_dataset, cls0_sample_datasets=cls0_sample_datasets)
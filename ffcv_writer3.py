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
# parser.add_argument('--observe_classes', type=list, default=[0,1,2,3,4,5,6,7,8,9], help='class')
parser.add_argument('--observe_classes', type=list, default=[0], help='class')
args = parser.parse_args()
device = 'cpu'

# 定义钩子函数
def hook(module, input, output):
    global last_conv_output
    last_conv_output = output
    
def label_to_prob(images, labels, cls):
    # 确保图像是 PyTorch 张量并且维度顺序是 (N, C, H, W)
    if isinstance(images, np.ndarray):
        images = torch.tensor(images).permute(0, 3, 1, 2).float()
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels).long()

    # 创建 DataLoader
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # 加载预训练的ResNet18模型
    # model = ResNet18(num_classes=10)  # 指定类别数为10
    # model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.load_state_dict(torch.load('ResNet18_Cifar10_95.46/checkpoint/resnet18_cifar10.pt', map_location=device))
    model = torch.load("data/blend/0.1/models/model_epoch_100.pth")
    # model.to(device)
    model.to(torch.device('cuda'))
    model.eval()
    all_probabilities = []

    # 获取模型输出
    with torch.no_grad():
        for images, labels in loader:
            # images = images.to(device)
            images = images.to(torch.device('cuda'))
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

    # 将所有概率拼接成一个数组
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    # test_trainset(model)
    # 找出 all_probabilities 中 argmax 不等于 cls 的元素
    incorrect_indices = np.argmax(all_probabilities, axis=1) != cls
    incorrect_count = np.sum(incorrect_indices)
    print(f"Number of elements with argmax not equal to cls {cls}: {incorrect_count}")
    print(f"Percentage of incorrect predictions: {incorrect_count / len(all_probabilities) * 100:.2f}%")
    # 将这些元素设置为 cls 的 one-hot 向量
    all_probabilities[incorrect_indices] = np.eye(all_probabilities.shape[1])[cls]
    print("all_probabilities.shape:", all_probabilities.shape)
    print("label:", torch.argmax(torch.tensor(all_probabilities), dim=1))
    return all_probabilities

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
        label_probs = label_to_prob(torch.tensor(observe_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                                    torch.tensor(observe_label_npy, dtype=torch.float32, device=device), cls)
        # np.save(f"{args.output_path}/label_probs_{cls}_backdoor.npy", label_probs)
        sample_dataset = TensorDataset(
            torch.tensor(observe_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            torch.tensor(label_probs, dtype=torch.float32, device=device))
        sample_datasets.append(sample_dataset)
        if cls == 0:
            backdoor_data_npy, backdoor_label_npy = observe_data_npy[:5000], label_probs[:5000]
            cls0_clean_data_npy, cls0_clean_label_npy = observe_data_npy[5000:], label_probs[5000:]
            np.save(f"{args.output_path}/backdoor_label_npy.npy", backdoor_label_npy)
            np.save(f"{args.output_path}/cls0_clean_label_npy.npy", cls0_clean_label_npy)
            backdoor_dataset = TensorDataset(
                torch.tensor(backdoor_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                torch.tensor(backdoor_label_npy, dtype=torch.float32, device=device))
            cls0_clean_dataset = TensorDataset(
                torch.tensor(cls0_clean_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                torch.tensor(cls0_clean_label_npy, dtype=torch.float32, device=device))
            total_sample_size = 4500
            backdoor_sample_size = int(total_sample_size * (5000 / 9500))  # 约2632个样本
            clean_sample_size = total_sample_size - backdoor_sample_size    # 约2368个样本
            
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
    return train_dataset, test_dataset, sample_datasets, backdoor_dataset, cls0_clean_dataset, cls0_sample_datasets

def write_datasets(train_dataset=None, test_dataset=None, sample_datasets=None, backdoor_dataset=None, cls0_clean_dataset=None, cls0_sample_datasets=None):
    if args.dataset == 'sample_dataset' or args.dataset == 'all':
        for idx, sample_dataset in enumerate(sample_datasets):
            class_write_path = f"{args.output_path}/train_data_class_{args.observe_classes[idx]}.beton"
            writer = DatasetWriter(class_write_path, {
                'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                'label': TorchTensorField(dtype=torch.float32, shape=(10,))  # 修改为保存概率
                # 'label': IntField()
            })
            writer.from_indexed_dataset(sample_dataset)
            if idx == 0:
                backdoor_write_path = f"{args.output_path}/train_data_class_0_backdoor.beton"
                writer = DatasetWriter(backdoor_write_path, {
                    'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                    'label': TorchTensorField(dtype=torch.float32, shape=(10,))  # 修改为保存概率
                })
                writer.from_indexed_dataset(backdoor_dataset)
                clean_write_path = f"{args.output_path}/train_data_class_0_clean.beton"
                writer = DatasetWriter(clean_write_path, {
                    'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                    'label': TorchTensorField(dtype=torch.float32, shape=(10,))  # 修改为保存概率
                })
                writer.from_indexed_dataset(cls0_clean_dataset)
                cls0_sample_write_path = f"{args.output_path}/train_data_class_0_sample.beton"
                writer = DatasetWriter(cls0_sample_write_path, {
                    'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                    'label': TorchTensorField(dtype=torch.float32, shape=(10,))  # 修改为保存概率
                })
                writer.from_indexed_dataset(cls0_sample_datasets)
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
    train_dataset, test_dataset, sample_datasets, backdoor_dataset, cls0_clean_dataset, cls0_sample_datasets = create_datasets()
    if args.dataset == 'all':
        write_datasets(train_dataset, test_dataset, sample_datasets, backdoor_dataset, cls0_clean_dataset, cls0_sample_datasets)
    elif args.dataset == 'train_dataset':
        write_datasets(train_dataset=train_dataset)
    elif args.dataset == 'test_dataset':
        write_datasets(test_dataset=test_dataset)
    elif args.dataset == 'sample_dataset':
        write_datasets(sample_datasets=sample_datasets, backdoor_dataset=backdoor_dataset, cls0_clean_dataset=cls0_clean_dataset, cls0_sample_datasets=cls0_sample_datasets)
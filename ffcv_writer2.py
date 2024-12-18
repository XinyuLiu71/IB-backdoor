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
parser.add_argument('--sampling_datasize', type=int, default=3000, help='sampling_datasize')
parser.add_argument('--observe_classes', type=list, default=[0,1,2,3,4,5,6,7,8,9], help='class')
args = parser.parse_args()
device = 'cpu'

def test_trainset(model):
    test_dataloader_path = "data/badnet_/0.1/badnet_10%.npz"

    # 加载数据集
    data = np.load(test_dataloader_path)
    test_images = data['arr_0']
    test_labels = data['arr_1']

    # 确保图像是3通道的
    if test_images.shape[-1] == 1:
        test_images = np.repeat(test_images, 3, axis=-1)

    # 将数据转换为 PyTorch 张量
    test_images = torch.tensor(test_images).permute(0, 3, 1, 2).float()  # 转换为 (N, C, H, W) 格式
    test_labels = torch.tensor(test_labels).long()
    # 创建 DataLoader
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    # 初始化变量以累积正确预测的数量和总样本数量
    correct_predictions = 0
    total_samples = 0

    # 获取模型输出
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # 如果需要，可以将输出转换为概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # 获取预测的类别
            _, predicted_classes = torch.max(outputs, 1)
            # 累积正确预测的数量和总样本数量
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

    # 计算并打印整个测试集的准确率
    accuracy = correct_predictions / total_samples
    print("Overall test set accuracy:", accuracy)

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
    model = torch.load("results/ob_infoNCE_10_06_0.1/best_model.pth")
    model.to(device)
    model.eval()
    all_probabilities = []

    # 获取模型输出
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
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
    # for cls in args.observe_classes:
    #     observe_data_npy = training_data_npy['arr_0'][training_data_npy['arr_1'] == cls]
    #     observe_label_npy = training_data_npy['arr_1'][training_data_npy['arr_1'] == cls]
    #     print(f"cls_{cls}.len = {len(observe_label_npy)}")

    #     data_label_pairs = list(zip(observe_data_npy, observe_label_npy))
    #     random.shuffle(data_label_pairs)
    #     train_data_label1_shuffled, train_label_label1_shuffled = zip(*data_label_pairs)
    #     train_data_label1_sampled = np.array(random.sample(train_data_label1_shuffled, args.sampling_datasize))
    #     train_label_label1_sampled = np.array(random.sample(train_label_label1_shuffled, args.sampling_datasize))
    #     if cls == 0:
    #         image_shuffle, label_shuffle = get_Sample(args.sampling_datasize, args.train_cleandata_path,
    #                                                 args.train_poisondata_path)
    #         label_probs = label_to_prob(torch.tensor(image_shuffle, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    #                                     torch.tensor(label_shuffle, dtype=torch.float32, device=device), cls)
    #         sample_dataset = TensorDataset(
    #             torch.tensor(image_shuffle, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    #             torch.tensor(label_probs, dtype=torch.float32, device=device))
    #     else:
    #         label_probs = label_to_prob(torch.tensor(train_data_label1_sampled, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    #                                     torch.tensor(train_label_label1_sampled, dtype=torch.float32, device=device), cls)
    #         sample_dataset = TensorDataset(
    #             torch.tensor(train_data_label1_sampled, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    #             torch.tensor(label_probs, dtype=torch.float32, device=device))
    #     sample_datasets.append(sample_dataset)
    return train_dataset, test_dataset, sample_datasets

def write_datasets(train_dataset=None, test_dataset=None, sample_datasets=None):
    if args.dataset == 'sample_dataset' or args.dataset == 'all':
        for idx, sample_dataset in enumerate(sample_datasets):
            class_write_path = f"{args.output_path}/observe_data_class_{args.observe_classes[idx]}.beton"
            writer = DatasetWriter(class_write_path, {
                'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
                'label': TorchTensorField(dtype=torch.float32, shape=(10,))  # 修改为保存概率
                # 'label': IntField()
            })
            writer.from_indexed_dataset(sample_dataset)
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

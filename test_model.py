import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from model.resnet import resnet18
from model.TNet import TNet
import torch.nn.functional as F
import math
import os
import random
import setproctitle
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder

# 检查 CUDA 设备的可用性
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.current_device()}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

# 加载模型（添加 map_location 参数以避免设备不匹配问题）
model = torch.load('results/ob_infoNCE_09_15_badnet_0.1/models.pth', map_location=device)

# 如果模型是用 DataParallel 封装的，移除封装
if isinstance(model, nn.DataParallel):
    model = model.module

model.to(device)
model.eval()

batch_size = 128

# Data decoding and augmentation
image_pipeline = [ToTensor(), ToDevice(device)]
label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

# Pipeline for each data field
pipelines = {
    'image': image_pipeline,
    'label': label_pipeline
}
num_workers = 0  # 尝试将 num_workers 设置为 0

# Load sample data for each class
dataloader_path = "data/badnet_/0.1/test_data.beton"
dataloader = Loader(dataloader_path, batch_size=batch_size, num_workers=num_workers,
                            order=OrderOption.RANDOM, pipelines=pipelines)

# 初始化用于跟踪每个类别准确率的变量
num_classes = 10  # 假设有10个类别，根据实际情况调整
class_correct = [0] * num_classes
class_total = [0] * num_classes



def test_model_on_data(model, dataloader, num_classes):
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    correct = 0
    size = dataloader.batch_size
    num_batches = len(dataloader)
    total = size * num_batches

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # 确保数据也在同一个设备上
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            c = (predicted == y).squeeze()
            for i in range(len(y)):
                label = y[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    overall_accuracy = 100 * correct / total
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy of class {i}: {class_acc:.2f}%')
        else:
            print(f'Accuracy of class {i}: N/A (no samples)')


def calculate_asr(model, dataloader, target_class, device):
    attack_success_count = 0
    total_triggered_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            attack_success_count += (predicted == target_class).sum().item()
            total_triggered_samples += y.size(0)

    asr = 100 * attack_success_count / total_triggered_samples
    print(f"Attack Success Rate (ASR): {asr:.2f}%")


# Load sample data for each class
dataloader_path = "data/badnet_/0.1/test_data.beton"
dataloader = Loader(dataloader_path, batch_size=batch_size, num_workers=num_workers,
                            order=OrderOption.RANDOM, pipelines=pipelines)

# Test on original data
print("Testing on original testset:")
test_model_on_data(model, dataloader, num_classes)

# Load and test on poison_data.npz
poison_data = np.load("data/badnet_/0.1/poisoned_test_data.npz")
poison_dataset = TensorDataset(
    torch.tensor(poison_data['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(poison_data['arr_1'], dtype=torch.long, device=device)
)
poison_dataloader = DataLoader(poison_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print("Testing on poison data:")
test_model_on_data(model, poison_dataloader, num_classes)

# Calculate ASR for poison data
target_class = 0  # Replace with the actual target class for the attack
calculate_asr(model, poison_dataloader, target_class, device)

# Load and test on clean_data.npz
# clean_data = np.load("data/badnet_/0.1/all_clean_data.npz")
# clean_dataset = TensorDataset(
#     torch.tensor(clean_data['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
#     torch.tensor(clean_data['arr_1'], dtype=torch.long, device=device)
# )
# clean_dataloader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# print("Testing on clean data:")
# test_model_on_data(model, clean_dataloader, num_classes)

def calculate_asr(model, dataloader, target_class, device):
    attack_success_count = 0
    total_triggered_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            attack_success_count += (predicted == target_class).sum().item()
            total_triggered_samples += y.size(0)

    asr = 100 * attack_success_count / total_triggered_samples
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
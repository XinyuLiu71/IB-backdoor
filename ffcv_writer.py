import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.resnet import resnet18
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, NDArrayDecoder
from model.TNet import TNet
import torch.nn.functional as F
import numpy as np
import math
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, TorchTensorField
import os
import setproctitle
from typing import List
import argparse
from ffcv.transforms.common import Squeeze

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, default='data/badNet_data_5.npz', help='Path to the training data')
parser.add_argument('--test_data_path', type=str, default='data/clean_new_testdata.npz', help='Path to the test data')
parser.add_argument('--output_path', type=str, default='test_dataset.beton', help='Path to the output .beton file')
parser.add_argument('--dataset', type=str, default='train_dataset',
                    help='Three types: train_dataset, test_dataset or observe_data')
args = parser.parse_args()
device = 'cpu'

training_data_npy = np.load(args.train_data_path)
test_data_npy = np.load(args.test_data_path)

train_dataset = TensorDataset(
    torch.tensor(training_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(training_data_npy['arr_1'], dtype=torch.long, device=device))
test_dataset = TensorDataset(
    torch.tensor(test_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(test_data_npy['arr_1'], dtype=torch.long, device=device))

# 提取标签为0的训练数据
observe_data = training_data_npy['arr_0'][training_data_npy['arr_1'] == 0]
observe_label = training_data_npy['arr_1'][training_data_npy['arr_1'] == 0]

# 创建TensorDataset
observe_data = TensorDataset(
    torch.tensor(observe_data, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(observe_label, dtype=torch.long, device=device))

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

write_path = args.output_path

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
    'label': IntField()
})

# Write dataset
writer.from_indexed_dataset(args.dataset)
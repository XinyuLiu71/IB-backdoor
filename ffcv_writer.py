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
from ffcv.transforms.common import Squeeze

device = 'cpu'

training_data_npy = np.load('data/badNet_data_5.npz')
test_data_npy = np.load('data/clean_new_testdata.npz')

train_dataset = TensorDataset(
    torch.tensor(training_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(training_data_npy['arr_1'], dtype=torch.long, device=device))
test_dataset = TensorDataset(
    torch.tensor(test_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(test_data_npy['arr_1'], dtype=torch.long, device=device))

# 提取标签为0的训练数据
train_data_label1 = training_data_npy['arr_0'][training_data_npy['arr_1'] == 0]
train_label_label1 = training_data_npy['arr_1'][training_data_npy['arr_1'] == 0]

# 创建TensorDataset
train_dataset_label1 = TensorDataset(
    torch.tensor(train_data_label1, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(train_label_label1, dtype=torch.long, device=device))


from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField


write_path = 'test_dataset.beton'

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': TorchTensorField(dtype = torch.float32, shape = (3, 32, 32)),
    'label': IntField()
})

# Write dataset
writer.from_indexed_dataset(test_dataset)
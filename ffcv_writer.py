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
parser.add_argument('--train_data_path', type=str, default='data/badNet_data.npz', help='Path to the training data')
parser.add_argument('--test_data_path', type=str, default='data/clean_new_testdata.npz', help='Path to the test data')
parser.add_argument('--output_path', type=str, default='test_dataset.beton', help='Path to the output .beton file')
parser.add_argument('--dataset', type=str, default='train_dataset',
                    help='Three types: train_dataset, test_dataset or sample_dataset')
parser.add_argument('--sampling_datasize', type=int, default=4000, help='sampling_datasize')
parser.add_argument('--observe_class', type=int, default=0, help='class')
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
observe_data_npy = training_data_npy['arr_0'][training_data_npy['arr_1'] == args.observe_class]
observe_label_npy = training_data_npy['arr_1'][training_data_npy['arr_1'] == args.observe_class]

# 创建TensorDataset
observe_data = TensorDataset(
    torch.tensor(observe_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(observe_label_npy, dtype=torch.long, device=device))

data_label_pairs = list(zip(observe_data_npy, observe_label_npy))
random.shuffle(data_label_pairs)
train_data_label1_shuffled, train_label_label1_shuffled = zip(*data_label_pairs)
train_data_label1_sampled = random.sample(train_data_label1_shuffled, args.sampling_datasize)
train_label_label1_sampled = np.array(random.sample(train_label_label1_shuffled, args.sampling_datasize))
# train_data_label1_sampled = np.array(train_data_label1_sampled)
sample_dataset = TensorDataset(
    torch.tensor(train_data_label1_sampled, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
    torch.tensor(train_label_label1_sampled, dtype=torch.long, device=device))
# sample_data = DataLoader(sample_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


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
if args.dataset == 'sample_dataset':
    writer.from_indexed_dataset(sample_dataset)
elif args.dataset == 'train_dataset':
    writer.from_indexed_dataset(train_dataset)
elif args.dataset == 'test_dataset':
    writer.from_indexed_dataset(test_dataset)
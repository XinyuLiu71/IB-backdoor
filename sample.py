import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.resnet import ResNet18
from model.TNet import TNet
import torch.nn.functional as F
import numpy as np
import math
import os
import random
import setproctitle
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder


def get_Sample(sample_size, clean_path, poison_path):
    clean_npy = np.load(clean_path)
    poison_npy = np.load(poison_path)

    fileRead_total_size = len(clean_npy['arr_0']) + len(poison_npy['arr_0'])
    percent = len(poison_npy['arr_0']) / fileRead_total_size

    poison_count = int(sample_size * percent)
    clean_count = sample_size - poison_count

    if sample_size > fileRead_total_size:
        print("total_size is larger than the total size of clean and poison data")
        return

    clean_image_npy = list(clean_npy['arr_0'])
    clean_label_npy = list(clean_npy['arr_1'])
    # print(clean_label_npy)
    clean_pairs = list(zip(clean_image_npy, clean_label_npy))
    random.shuffle(clean_pairs)
    clean_pairs = clean_pairs[:clean_count]

    poison_image_npy = list(poison_npy['arr_0'])
    poison_label_npy = list(poison_npy['arr_1'])
    poison_pairs = list(zip(poison_image_npy, poison_label_npy))
    random.shuffle(poison_pairs)
    poison_pairs = poison_pairs[:poison_count]

    clean_pairs.extend(poison_pairs)
    samplePairs = clean_pairs
    image_shuffle, label_shuffle = zip(*samplePairs)


    return image_shuffle, label_shuffle, poison_count

# image_shuffle, label_shuffle = get_Sample(5000, "data/clean_0.9.npz", "./data/poison_0.1.npz")
# print(label_shuffle)

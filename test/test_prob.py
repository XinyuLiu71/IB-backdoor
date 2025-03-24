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
from ffcv.transforms import ToTensor, ToDevice, Squeeze, RandomHorizontalFlip, RandomResizedCrop, RandomBrightness, RandomContrast, RandomSaturation
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data decoding and augmentation
image_pipeline = [ToTensor(), ToDevice(device)]
label_pipeline_sample = [ToTensor(), ToDevice(device)]

# Pipeline for each data field
pipelines_sample = {
    'image': image_pipeline,
    'label': label_pipeline_sample
}
sample_dataloader_path = f"data/badnet_/0.1/observe_data_class_0.beton"
sample_dataloader = Loader(sample_dataloader_path, batch_size=256, num_workers=16,
                                        order=OrderOption.RANDOM, pipelines=pipelines_sample)
# Get all data from the data loader
for batch, (images, labels) in enumerate(sample_dataloader):
    labels = torch.argmax(labels, dim=1)
    print(labels)
    break
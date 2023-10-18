# from dataset import PoisonedCIFAR10
from datasets import *
import torch
import torchvision
from typing import List

import argparse

from ffcv.fields import IntField, RGBImageField, TorchTensorField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
import random

import os
from pathlib import Path


def write_dataset(args):
    if args.dataset == "cifar10":

        DATASET = {
            'Badnet': {'class': BadnetCIFAR10, 'type': 'RGB'},
            'Blend': {'class': BlendCIFAR10, 'type': 'RGB'},
            'Wanet': {'class': WaNetCIFAR10, 'type': 'torch'},
            'CleanLabel': {'class': CleanLabelCIFAR10, 'type': 'torch'},
            'DynamicBackdoor': {'class': DynamicBackdoorCIFAR10, 'type': 'torch'},
            'SIG': {'class': SIGCIFAR10, 'type': 'RGB'},
            'LabelConsistent': {'class': LabelConsistentCIFAR10, 'type': 'RGB'},
            'Trojan': {'class': TrojanCIFAR10, 'type': 'RGB'},
            'ISSBA': {'class': ISSBAImagenet, 'type': 'RGB'},
            'AdaptiveBlend': {'class': AdapBlendCIFAR10, 'type': 'RGB'},
            'DFST': {'class': DFSTCIFAR10, 'type': 'RGB'},
            'Badnet_Adaptive2': {'class': BadnetCIFAR10_Adaptive2, 'type': 'RGB'},
            'Badnet_1to1': {'class': BadnetCIFAR10_1to1, 'type': 'RGB'},
            'Badnet_Adaptive3': {'class': BadnetCIFAR10_Adaptive3, 'type': 'RGB'},
            'Badnet_Adaptive1': {'class': BadnetCIFAR10_Adaptive1, 'type': 'RGB'},
            'Badnet_Adaptive4': {'class': BadnetCIFAR10_Adaptive4, 'type': 'RGB'},
            'Badnet_allto1': {'class': BadnetCIFAR10_allto1, 'type': 'RGB'},
        }
        img_size = 32

    elif args.dataset == "imagenet200":

        DATASET = {
            'Badnet': {'class': BadnetImagenet200, 'type': 'RGB'},
            'Blend': {'class': BlendImagenet200, 'type': 'RGB'},
            'Wanet': {'class': WaNetImagenet200, 'type': 'torch'},
            'CleanLabel': {'class': CleanLabelImagenet200, 'type': 'torch'},
            # 'LabelConsistent' : {'class': LabelConsistentImagenet200, 'type': 'RGB'},
            # 'Trojan' : {'class':TrojanImagenet200, 'type': 'RGB'},
        }
        img_size = 224

    elif args.dataset == "tinyimagenet":

        DATASET = {
            'Badnet': {'class': BadnetTinyimagenet, 'type': 'RGB'},
            'Blend': {'class': BlendTinyimagenet, 'type': 'RGB'},
            'Wanet': {'class': WaNetTinyimagenet, 'type': 'torch'},
            'CleanLabel': {'class': CleanLabelTinyimagenet, 'type': 'torch'},
            # 'LabelConsistent' : {'class': LabelConsistentTinyimagenet, 'type': 'RGB'},
            # 'Trojan' : {'class':TrojanTinyimagenet, 'type': 'RGB'},
        }
        img_size = 64

    datasets = {
        'train': DATASET[args.attack]['class']('data', train=True, poison_ratio=args.poison_ratio, target=args.target,
                                               partition='None'),  # upper_right=True
        'test_clean': DATASET[args.attack]['class']('data', train=False, poison_ratio=0, target=args.target),
        'test_poison': DATASET[args.attack]['class']('data', train=False, poison_ratio=1, target=args.target,
                                                     asr_calc=True)
    }

    if args.save_samples == 'True':
        path = f'Results/Saved_images/{args.dataset}/{args.attack}/{args.poison_ratio}'
        if not os.path.isdir(path):
            Path(path).mkdir(parents=True)
        datasets['train'].save_images(path)

    if DATASET[args.attack]['type'] == 'torch':
        imagewriter = TorchTensorField(torch.float32, (3, img_size, img_size))
    elif DATASET[args.attack]['type'] == 'RGB':
        imagewriter = RGBImageField()

    for (name, ds) in datasets.items():
        if args.target == 1:
            pathwriter = f'data/{args.dataset}_{args.attack}_{args.poison_ratio}_{name}.beton'
        else:
            pathwriter = f'data/{args.dataset}_{args.attack}_{args.target}_{args.poison_ratio}_{name}.beton'
        writer = DatasetWriter(pathwriter, {
            'image': imagewriter,
            'label': IntField(),
            'index': IntField(),
            'poisonlabel': IntField()
        })
        writer.from_indexed_dataset(ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##################################### general setting #################################################
    parser.add_argument('--data', type=str, default='../data0', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--arch', type=str, default='res18', help='model architecture')
    parser.add_argument('--expnumber', type=str, default='Bilevel_Optim_highscale', help='Experiment Number')

    ##################################### training setting ################################################
    parser.add_argument('--poison_ratio', default=0.1, type=float, help='Poison Ratio')
    parser.add_argument('--attack', type=str, help='Give attack name')
    parser.add_argument('--save_samples', type=str, default='False', help='Give attack name')
    parser.add_argument('--target', default=1, type=int, help='Target label')

    opt = parser.parse_args()
    write_dataset(opt)










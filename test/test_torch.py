import torch
import torchvision
from torchvision import datasets, transforms

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
dataset = datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform
        )
dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, num_workers=2
        )
dataiter = iter(dataloader)
images, labels = dataiter.__next__()
print(images.shape)
print(images.min(), images.max())
print(labels)
# torch.Size([4, 3, 32, 32])
# tensor(-1.) tensor(1.)
# tensor([6, 9, 8, 4])

from tensorflow.keras.datasets import cifar10
import numpy as np

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0

print(train_images.shape)
print(train_images.min(), train_images.max())
# (50000, 32, 32, 3)
# 0.0 1.0
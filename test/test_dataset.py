import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, RandomHorizontalFlip, RandomResizedCrop, RandomBrightness, RandomContrast, RandomSaturation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 检查 poison_data.npz
# poison_data = np.load("data/badnet_/0.01/poison_data.npz")
# poison_dataset = TensorDataset(
#     torch.tensor(poison_data['arr_0'], dtype=torch.float32).permute(0, 3, 1, 2).to(device),
#     torch.tensor(poison_data['arr_1'], dtype=torch.long).to(device)
# )
# poison_dataloader = DataLoader(poison_dataset, batch_size=256, shuffle=False, num_workers=0)
# for X, y in poison_dataloader:
#     unique, counts = np.unique(y.cpu().numpy(), return_counts=True)  # 将张量移动到 CPU 并转换为 NumPy 数组
#     print("Poison data class distribution:", dict(zip(unique, counts)))

# # 检查 clean_data.npz
# clean_data = np.load("data/badnet_/0.01/clean_data.npz")
# clean_dataset = TensorDataset(
#     torch.tensor(clean_data['arr_0'], dtype=torch.float32).permute(0, 3, 1, 2).to(device),
#     torch.tensor(clean_data['arr_1'], dtype=torch.long).to(device)
# )
# clean_dataloader = DataLoader(clean_dataset, batch_size=256, shuffle=False, num_workers=0)
# for X, y in clean_dataloader:
#     unique, counts = np.unique(y.cpu().numpy(), return_counts=True)  # 将张量移动到 CPU 并转换为 NumPy 数组
#     print("Clean data class distribution:", dict(zip(unique, counts)))

# =================================================================================================
# 检查 train_dataset.npz
test_data = np.load("data/cifar10/badnet/0.1/badnet_0.1.npz")
test_images = torch.tensor(test_data['arr_0'], dtype=torch.float32).permute(0, 3, 1, 2).to(device)
test_labels = torch.tensor(test_data['arr_1'], dtype=torch.long).to(device)

# image_pipeline = [
#         ToTensor(), 
#         ToDevice(device)
#     ]

# label_pipeline = [ToTensor(), ToDevice(device)]

#     # Pipeline for each data field
# pipelines = {
#         'image': image_pipeline,
#         'label': label_pipeline
#     }
# test_dataloader_path = "data/badnet/0.1/train_data_class_2.beton"
# test_dataloader = Loader(test_dataloader_path, batch_size=256, num_workers=16,
#                             order=OrderOption.RANDOM, pipelines=pipelines)
# # 从数据加载器中获取所有数据
# test_images = []
# test_labels = []

# for batch, (images, labels) in enumerate(test_dataloader):
#     test_images.append(images)
#     test_labels.append(labels)

# test_images = torch.cat(test_images)
# test_labels = torch.cat(test_labels)

# 打印数据集信息
print(f"Total number of samples: {len(test_labels)}")
print(f"Image shape: {test_images.shape}")
print(f"Labels shape: {test_labels.shape}")

# 计算每个类别的样本数量
unique, counts = np.unique(test_labels.cpu().numpy(), return_counts=True)
class_distribution = dict(zip(unique, counts))

# 抽取每个类别1%的样本
sampled_images = []
sampled_labels = []
for label in unique:
    label_indices = (test_labels == label).nonzero(as_tuple=True)[0]
    sample_size = max(1, int(0.01 * class_distribution[label]))  # 至少抽取1个样本
    sampled_indices = label_indices[torch.randperm(len(label_indices))[:sample_size]]
    sampled_images.append(test_images[sampled_indices])
    sampled_labels.append(test_labels[sampled_indices])

sampled_images = torch.cat(sampled_images)
sampled_labels = torch.cat(sampled_labels)

# 展示样本图片
def imshow_grid(images, labels, title):
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(title)
    
    # 如果只有一个轴，确保 axes 是一个二维数组
    if grid_size == 1:
        axes = np.array([[axes]])
    elif grid_size > 1 and len(axes.shape) == 1:
        axes = axes.reshape((grid_size, grid_size))
    
    print(images.max(), images.min())
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # img = images[i] / 2 + 0.5  # unnormalize
            npimg = images[i].cpu().numpy()
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(f'Label: {labels[i].item()}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.savefig(f"sample1%_class{labels[0].item()}.png")

# 展示每个类别的样本图片，总共展示10张图
for label in unique[:10]:  # 只展示前10个类别
    label_indices = (sampled_labels == label).nonzero(as_tuple=True)[0]
    imshow_grid(sampled_images[label_indices], sampled_labels[label_indices], f'Class {label} Samples')
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, RandomHorizontalFlip, RandomResizedCrop, RandomBrightness, RandomContrast, RandomSaturation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attack_type = "wanet"
test_dataloader_path = f"data/{attack_type}/0.1/poisoned_test_data.npz"

# 加载数据集
data = np.load(test_dataloader_path)
images = data['arr_0']
labels = data['arr_1']


# 选择标签为0的数据
class0_indices = np.where(labels == 0)[0]
class0_images = images[class0_indices]

# 获取class0中的第一张图像
# 41 label_consistent
first_class0_image = class0_images[71]

# 展示class0中的第一张图像
plt.imshow(first_class0_image)
plt.axis('off')  # 移除刻度
plt.savefig(f"image/class0_{attack_type}.png")

# 选择标签为0的数据
# class0_indices = np.where(labels == 0)[0]
# class0_images = images[class0_indices]

# # 展示class0中的前十张图像
# fig, axes = plt.subplots(5, 10, figsize=(15, 6))
# for i, ax in enumerate(axes.flat):
#     if i < len(class0_images):
#         ax.imshow(class0_images[i+30])
#         ax.axis('off')  # 移除刻度
# plt.savefig(f"image/class0_{attack_type}_top10.png")
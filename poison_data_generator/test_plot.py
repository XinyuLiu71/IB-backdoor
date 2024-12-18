import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('../data/badnet_/0.1/test_data.npz')
images = data['arr_0']  # 假设数据包含图像
labels = data['arr_1']  # 假设数据包含标签

# 归一化图像数据到 [0, 1] 范围
# images = (images - images.min()) / (images.max() - images.min())
print(f"max: {images.max()}, min: {images.min()}")
# 获取唯一的类别
unique_labels = np.unique(labels)

# 每个类别采样10个
sampled_images = []
sampled_labels = []

for label in unique_labels:
    indices = np.where(labels == label)[0]
    sampled_indices = np.random.choice(indices, 10, replace=False)
    sampled_images.extend(images[sampled_indices])
    sampled_labels.extend(labels[sampled_indices])

# 绘制图像
fig, axes = plt.subplots(len(unique_labels), 10, figsize=(15, len(unique_labels) * 1.5))

for i, ax in enumerate(axes.flat):
    ax.imshow(sampled_images[i])  # 假设图像是灰度图
    ax.axis('off')
    ax.set_title(f'Label: {sampled_labels[i]}')

plt.tight_layout()
plt.savefig("sampled_test_images.png")

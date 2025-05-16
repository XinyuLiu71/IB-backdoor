import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = np.load("data/cifar10/ftrojan/0.1/poisoned_test_data.npz")
# data = np.load("data/cifar10/badnet/0.1/badnet_0.1.npz")


test_images = torch.tensor(data['arr_0'], dtype=torch.float32).permute(0, 3, 1, 2).to(device)
test_labels = torch.tensor(data['arr_1'], dtype=torch.long).to(device)


# # 将图片数据转换回CPU并转换为numpy数组
# # images = test_images[3000:3500].cpu().numpy() # 3500, 4000
# # labels = test_labels[3000:3500].cpu().numpy()
# images = test_images[500:1000].cpu().numpy() # 3500, 4000
# labels = test_labels[500:1000].cpu().numpy()

# # 创建网格显示图片
# n_images = 500
# n_cols = 10
# n_rows = (n_images + n_cols - 1) // n_cols

# plt.figure(figsize=(20, 2*n_rows))
# for i in range(n_images):
#     plt.subplot(n_rows, n_cols, i+1)
#     # 转换图片格式从(C,H,W)到(H,W,C)
#     img = images[i].transpose(1, 2, 0)
#     # 归一化到[0,1]范围
#     img = (img - img.min()) / (img.max() - img.min())
#     plt.imshow(img)
#     plt.title(f'Label: {labels[i]}')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()
# plt.savefig('cifar10_adaptive_blend_0.1.png', dpi=300)

# 获取第678张图片
image = test_images[678].cpu().numpy()
label = test_labels[678].cpu().numpy()

# 创建单个图片的显示
plt.figure(figsize=(5, 5))
# 转换图片格式从(C,H,W)到(H,W,C)
img = image.transpose(1, 2, 0)
# 归一化到[0,1]范围
img = (img - img.min()) / (img.max() - img.min())
plt.imshow(img)
# plt.title(f'Label: {label}')
plt.axis('off')

# 移除所有边距
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
plt.savefig('cifar10_ftrojan_0.1.png', dpi=300)

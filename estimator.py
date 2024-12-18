      
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.TNet import TNet
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_DV(T, Y, Z_, t):
    ema_rate = 0.01
    e_t2_ema = None
    t2 = T(Y, Z_)
    e_t2 = t2.exp()
    e_t2 = e_t2.clamp(max=1e20)
    e_t2_mean = e_t2.mean()
    if e_t2_ema is None:
        loss = -(t.mean() - e_t2_mean.log())
        e_t2_ema = e_t2_mean
    else:
        """
        log(e_t2_mean)' = 1/e_t2_mean * e_t2_mean'
        e_t2_mean' = sum(e_t2')/b
        e_t2' = e_t2 * t2'
        """
        e_t2_ema = (1 - ema_rate) * e_t2_ema + ema_rate * e_t2_mean
        loss = -(t.mean() - (t2 * e_t2.detach()).mean() / e_t2_ema.item())
        # loss = -(t.mean() - e_t2_mean / e_t2_ema.item())
    return t2, e_t2, loss


# def compute_infoNCE(T, Y, Z, t, num_negative_samples=256, batch_size=128):
#     total_samples = Y.shape[0]
#     t_positive = t.squeeze()
#     log_sum_exp_list = []
    
#     for i in range(0, total_samples, batch_size):
#         end = min(i + batch_size, total_samples)
#         Y_batch = Y[i:end]
#         t_positive_batch = t_positive[i:end]
        
#         # 随机选择负样本
#         negative_indices = torch.randint(0, total_samples, (end - i, num_negative_samples), device=Y.device)
#         Z_negative = Z[negative_indices]
        
#         # 计算负样本的得分
#         Y_expanded = Y_batch.unsqueeze(1).expand(-1, num_negative_samples, -1)
#         t_negative = T(Y_expanded.reshape(-1, Y.shape[1]), Z_negative.reshape(-1, Z.shape[1]))
#         t_negative = t_negative.view(end - i, num_negative_samples)
        
#         # 计算 InfoNCE loss（使用 LogSumExp 技巧来提高数值稳定性）
#         logits = torch.cat([t_positive_batch.unsqueeze(1), t_negative], dim=1)
#         log_sum_exp = logits.exp().mean(dim=1).log()
#         log_sum_exp_list.append(log_sum_exp)
    
#     log_sum_exp = torch.cat(log_sum_exp_list).mean()
#     loss = -t_positive.mean() + log_sum_exp
    
#     return log_sum_exp, loss

def compute_infoNCE(T, Y, Z, t, num_negative_samples=200):
    batch_size = Y.shape[0]
    
    # 随机选择负样本
    negative_indices = torch.randint(0, batch_size, (batch_size, num_negative_samples), device=Y.device)
    Z_negative = Z[negative_indices]
    
    # 计算正样本的得分
    t_positive = t.squeeze()
    # 计算负样本的得分
    Y_expanded = Y.unsqueeze(1).expand(-1, num_negative_samples, -1)
    t_negative = T(Y_expanded.reshape(-1, Y.shape[1]), Z_negative.reshape(-1, Z.shape[1]))
    t_negative = t_negative.view(batch_size, num_negative_samples)
    
    # 计算 InfoNCE loss
    logits = torch.cat([t_positive.unsqueeze(1), t_negative], dim=1)
    # 创建标签，正样本在第0位
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(Y.device)  # (batch_size,)

    # 使用交叉熵损失来计算 InfoNCE 损失
    loss = -math.log(num_negative_samples+1) + F.cross_entropy(logits, labels)
    
    return loss

# 生成Y和X数据
np.random.seed(42)
Y = np.random.normal(0, np.sqrt(3), (10000, 5))
noise = np.random.normal(0, np.sqrt(0.5), (10000, 3))
X = Y[:, :3] + noise

# 计算真实的互信息
cov_Y = np.cov(Y, rowvar=False)  # 计算Y的协方差矩阵
cov_X = np.cov(X, rowvar=False)  # 计算X的协方差矩阵
# 提取需要的子矩阵
cov_Y_3 = cov_Y[:3, :3]  # 只取Y的前三维的协方差矩阵
cov_X_Y = cov_Y[:3, :3] + np.cov(noise, rowvar=False)  # X的协方差矩阵 K_X
# 计算行列式
det_cov_X = np.linalg.det(cov_X_Y)
det_cov_Y_3 = np.linalg.det(cov_Y_3)
# 联合协方差矩阵的行列式计算
det_cov_joint = np.linalg.det(cov_Y_3) * np.linalg.det(cov_X_Y - cov_Y_3)
# 计算互信息
real_mi = 0.5 * np.log(det_cov_X * det_cov_Y_3 / det_cov_joint)
print(f"real mi : {real_mi}")


# 将数据转换为torch的tensor
Y_torch = torch.tensor(Y, dtype=torch.float32).to(device)
X_torch = torch.tensor(X, dtype=torch.float32).to(device)

# 初始化模型
T_infonce = TNet(in_dim=Y_torch.shape[1] + X_torch.shape[1], hidden_dim=512).to(device)
T_mine = TNet(in_dim=Y_torch.shape[1] + X_torch.shape[1], hidden_dim=512).to(device)

LR = 2e-4
optimizer_infonce = torch.optim.Adam(T_infonce.parameters(), lr=LR, weight_decay=1e-4)
optimizer_mine = torch.optim.Adam(T_mine.parameters(), lr=LR, weight_decay=1e-4)

# 添加学习率调度器
scheduler_infonce = ReduceLROnPlateau(optimizer_infonce, mode='max', factor=0.1, patience=10, verbose=True)
scheduler_mine = ReduceLROnPlateau(optimizer_mine, mode='max', factor=0.1, patience=10, verbose=True)

EPOCHS = 500
mi_values_infonce = []
mi_values_mine = []
real_mi_values = [real_mi] * EPOCHS  # 真实互信息在所有训练epoch中保持不变

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    Z_ = X_torch[torch.randperm(X_torch.size(0))]  # 生成打乱后的 Z_

    # InfoNCE训练
    optimizer_infonce.zero_grad()

    t_infonce = T_infonce(Y_torch, X_torch)
    loss_infonce = compute_infoNCE(T_infonce, Y_torch, X_torch, t_infonce)

    loss_infonce.backward()
    optimizer_infonce.step()
    
    # mi_infonce = t_infonce.mean().item() - t2.mean().item()
    mi_infonce = -loss_infonce.item()
    mi_values_infonce.append(mi_infonce)

    # 更新学习率
    scheduler_infonce.step(mi_infonce)

    # MINE训练
    optimizer_mine.zero_grad()

    t_mine = T_mine(Y_torch, X_torch)  # 计算联合分布样本的得分    
    _, e_t2_mine, loss_mine = compute_DV(T_mine, Y_torch, Z_, t_mine)

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(T_mine.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer_mine.step()

    mi_mine = t_mine.mean().item() - e_t2_mine.mean().log().item()
    mi_values_mine.append(mi_mine)

    # 更新学习率
    scheduler_mine.step(mi_mine)

    print(f'InfoNCE MI: {mi_infonce:.4f}, MINE MI: {mi_mine:.4f}')



# 绘制MI估计曲线
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, mi_values_infonce, label='InfoNCE MI')
plt.plot(epochs, mi_values_mine, label='MINE MI')
plt.plot(epochs, real_mi_values, label='Real MI', linestyle='--')
plt.xlabel('Training Epoch')
plt.ylabel('Mutual Information')
plt.title('Comparison of Mutual Information Estimation: InfoNCE vs MINE')
plt.legend()
# 保存图片到文件
plt.savefig('mi_estimation_comparison.png')

    
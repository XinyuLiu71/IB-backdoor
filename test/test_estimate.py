import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.neighbors import KernelDensity
import numpy as np
from model.TNet import TNet
from torch.nn import functional as F
def infoNCE_loss(Y, Z_pos, Z_neg, temperature=0.1):
    """
    计算 InfoNCE 损失。
    
    参数：
    Y: 输入表示，形状为 (batch_size, embedding_dim)
    Z_pos: 正样本表示，形状为 (batch_size, embedding_dim)
    Z_neg: 负样本表示，形状为 (batch_size, num_neg_samples, embedding_dim)
    temperature: 控制对比学习中 logits 的缩放
    
    返回：
    InfoNCE 损失值
    """
    # 正样本对的打分
    pos_score = T(Y, Z_pos) / temperature  # (batch_size, 1)
    
    # 负样本对的打分
    Y_expanded = Y.unsqueeze(1).expand(-1, Z_neg.size(1), -1)  # (batch_size, num_neg_samples, embedding_dim)
    neg_score = T(Y_expanded.reshape(-1, Y.size(-1)), Z_neg.reshape(-1, Z_neg.size(-1))).view(Y.size(0), -1) / temperature  # (batch_size, num_neg_samples)
    
    # 拼接正负样本对的分数
    logits = torch.cat([pos_score, neg_score], dim=1)  # (batch_size, 1 + num_neg_samples)
    
    # 计算 InfoNCE 损失
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=Y.device)  # 正样本的标签为 0
    loss = F.cross_entropy(logits, labels)
    
    return loss

# 示例数据
batch_size = 32
embedding_dim = 128
num_neg_samples = 10

# 随机初始化 Y, Z_pos, Z_neg 作为示例
Y = torch.randn(batch_size, embedding_dim)
Z_pos = torch.randn(batch_size, embedding_dim)  # 正样本对
Z_neg = torch.randn(batch_size, num_neg_samples, embedding_dim)  # 负样本对

T = TNet(in_dim=Y.shape[1] + Z_pos.shape[1], hidden_dim=128)
# 计算 InfoNCE 损失
loss = infoNCE_loss(Y, Z_pos, Z_neg)
print("InfoNCE loss:", loss.item())

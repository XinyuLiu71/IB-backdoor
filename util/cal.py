import torch
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy import stats

def get_acc(outputs, labels):
    """calculate acc"""
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0] * 1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num
    return acc

def calculate_asr(model, dataloader, target_class, device):
    model.eval()
    attack_success_count = 0
    total_triggered_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            # unique, counts = np.unique(predicted.cpu().numpy(), return_counts=True)
            # class_distribution = dict(zip(unique, counts))
            # print(class_distribution)
            attack_success_count += (predicted == target_class).sum().item()
            total_triggered_samples += y.size(0)

    asr = 100 * attack_success_count / total_triggered_samples
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    return asr

def compute_infoNCE(T, Y, Z, num_negative_samples=256):
    batch_size = Y.shape[0]
    # 随机选择负样本
    negative_indices = torch.randint(0, batch_size, (batch_size, num_negative_samples), device=Y.device)
    Z_negative = Z[negative_indices]
    
    # 计算正样本的得分
    t_positive = T(Y, Z).squeeze() # (batch_size, )
    # 计算负样本的得分
    Y_expanded = Y.unsqueeze(1).expand(-1, num_negative_samples, -1) # (batch_size, num_negative_samples, Y.dim)
    t_negative = T(Y_expanded.reshape(-1, Y.shape[1]), Z_negative.reshape(-1, Z.shape[1])) # (batch_size*num_negative_samples, )
    t_negative = t_negative.view(batch_size, num_negative_samples) # (batch_size, num_negative_samples)
    
    # 计算 InfoNCE loss
    logits = torch.cat([t_positive.unsqueeze(1), t_negative], dim=1).to(Y.device)  # (batch_size, num_negative_samples+1)
    log_sum_exp = logits.logsumexp(dim=1)
    # log_sum_exp = logits.exp().mean(dim=1).log()
    
    diffs = t_positive - log_sum_exp.mean() + math.log(num_negative_samples+1)
    loss = -diffs.mean()
    # loss = -diffs.mean()
    
    return loss, diffs

# Improved early stopping with better convergence criteria
def dynamic_early_stop(values, window=10, delta=1e-3, patience=5):
    if len(values) < window + patience:
        return False
    
    recent_values = values[-window:]
    slope, _, r_value, _, _ = stats.linregress(range(window), recent_values)
    
    # Check both the slope and the absolute change
    if abs(slope) < delta and max(recent_values) - min(recent_values) < delta:
        # Check if we're stuck at this plateau for 'patience' iterations
        if len(values) >= window + patience:
            plateau_values = values[-(window+patience):-window]
            plateau_range = max(plateau_values) - min(plateau_values)
            if plateau_range < delta:
                return True
    return False

def compute_class_accuracy(model, dataloader, num_classes):
    model.eval()
    correct = [0] * num_classes
    total = [0] * num_classes
    
    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(y)):
                label = y[i].item()
                total[label] += 1
                if predicted[i] == label:
                    correct[label] += 1
    
    accuracies = [100 * correct[i] / total[i] if total[i] > 0 else 0 for i in range(num_classes)]
    return accuracies

def analyze_sample_scores(sample_scores_dict, flag, output_dir, epoch):
    # 将所有类别的得分合并到一个列表中
    all_scores = []
    all_class_indices = []
    all_sample_indices = []
    
    for class_idx, scores in sample_scores_dict.items():
        all_scores.append(scores.cpu())  # 确保scores在CPU上
        all_class_indices.extend([class_idx] * len(scores))
        all_sample_indices.append(torch.arange(len(scores)))
    
    all_scores = torch.cat(all_scores)
    all_class_indices = torch.tensor(all_class_indices)
    all_sample_indices = torch.cat(all_sample_indices)
    
    # 计算所有得分的统计信息
    mean_score = all_scores.mean().item()
    std_score = all_scores.std().item()
    
    # 找出得分异常高的样本（例如，高于平均值两个标准差）
    threshold = mean_score + 2 * std_score
    suspicious_mask = all_scores > threshold
    suspicious_scores = all_scores[suspicious_mask]
    suspicious_classes = all_class_indices[suspicious_mask]
    suspicious_indices = all_sample_indices[suspicious_mask]
    
    print(f"Overall - Mean score: {mean_score:.4f}, Std: {std_score:.4f}")
    print(f"Number of suspicious samples: {len(suspicious_scores)}")
    
    # 保存可疑样本的信息
    suspicious_info = np.column_stack((suspicious_classes.cpu().detach().numpy(), 
                                       suspicious_indices.cpu().detach().numpy(), 
                                       suspicious_scores.cpu().detach().numpy()))
    np.save(os.path.join(output_dir, f'suspicious_samples_by_{flag}_epoch_{epoch}.npy'), suspicious_info)
    
    # 绘制每个类别的得分箱线图
    plt.figure(figsize=(12, 6))
    plt.boxplot([scores.cpu().detach().numpy() for scores in sample_scores_dict.values()], labels=sample_scores_dict.keys())
    plt.title(f'Score Distribution by {flag} - Epoch {epoch}')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.savefig(os.path.join(output_dir, f'score_distribution_by_{flag}_epoch_{epoch}.png'))
    plt.close()
    
    # 输出每个类别中可疑样本的数量
    for class_idx in sample_scores_dict.keys():
        class_suspicious_count = (suspicious_classes == class_idx).sum().item()
        print(f"Class {class_idx} - Number of suspicious samples: {class_suspicious_count}")

    return suspicious_info
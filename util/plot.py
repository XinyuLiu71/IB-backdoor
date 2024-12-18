import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
from openTSNE import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

def plot_and_save_mi(mi_values_dict, mode, output_dir, epoch):
    plt.figure(figsize=(12, 8))
    for class_idx, mi_values in mi_values_dict.items():
        if isinstance(class_idx, str):  # 对于 '0_backdoor', '0_clean' 和 '0_sample'
            if "backdoor" in class_idx:
                label = "Class 0 Backdoor"
            elif "clean" in class_idx:
                label = "Class 0 Clean"
            elif "sample" in class_idx:
                label = "Class 0 Sample"
            mi_values_np = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in mi_values]
            plt.plot(range(1, len(mi_values_np) + 1), mi_values_np, label=label)
        else:
            epochs = range(1, len(mi_values) + 1)
            mi_values_np = mi_values.cpu().numpy() if isinstance(mi_values, torch.Tensor) else mi_values
            if int(class_idx) == 0:
                plt.plot(epochs, mi_values_np, label=f'Class {class_idx}')
            else:
                plt.plot(epochs, mi_values_np, label=f'Class {class_idx}', linestyle='--')
    
    plt.xlabel('Epochs')
    plt.ylabel('MI Value')
    plt.title(f'MI Estimation over Epochs ({mode}) - Training Epoch {epoch}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'mi_plot_{mode}_epoch_{epoch}.png'))
    plt.close()

def plot_train_acc_ASR(train_accuracies, test_accuracies, ASR, epochs, outputs_dir):
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.plot(range(1, epochs + 1), ASR, label='ASR')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Training')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(outputs_dir + '/accuracy_plot.png')

def plot_train_loss_by_class(train_losses, epochs, num_classes, outputs_dir):
    plt.figure(figsize=(12, 8))
    for c in range(num_classes):
        plt.plot(range(1, epochs + 1), [losses[c] for losses in train_losses], label=f'Class {c}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss by Class over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outputs_dir, 'train_loss_by_class_plot.png'))
    plt.close()

def plot_tsne(t, labels, is_backdoor, epoch, outputs_dir, prefix='t'):
    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, n_jobs=16)
    # t_tsne = tsne.fit_transform(t.cpu().numpy())
    t_tsne = tsne.fit(t.cpu().numpy())

    # 计算指标
    silhouette_avg = silhouette_score(t_tsne, labels)
    davies_bouldin = davies_bouldin_score(t_tsne, labels)

    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))

    # 创建一个新的标签数组，将类别标签与是否是 backdoor 数据的标记结合
    # 将 backdoor 数据标记为 10（一个额外的类别），其他类别保留原标签
    combined_labels = labels.cpu().numpy().copy()
    combined_labels[is_backdoor.cpu().numpy() == 1] = 10  # 10 代表 backdoor 类别
    
    # 创建一个颜色映射，0-9 是类别，10 是 backdoor
    palette = sns.color_palette("tab10", n_colors=10)  # 使用 10 个颜色表示 10 个类别
    palette.append('red')  # 添加一个颜色来表示 backdoor 类
    
    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=t_tsne[:, 0], y=t_tsne[:, 1], hue=combined_labels, 
                    palette=palette, legend='full', marker='o')
    
    # 设置标题和轴标签
    plt.title(f't-SNE of {prefix} at Epoch {epoch}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # 自定义图例，确保 'backdoor' 的颜色是红色
    # 创建图例条目
    legend_labels = [f'Class {i}' for i in range(10)] + ['Backdoor']
    custom_lines = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in range(10)]
    # 添加 'Backdoor' 红色图例
    custom_lines.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10))

    # 添加图例
    plt.legend(handles=custom_lines, labels=legend_labels, title='Class')

    # 保存图像
    plt.savefig(os.path.join(outputs_dir, f'tsne_{prefix}_epoch_{epoch}.png'))
    plt.close()

    return silhouette_avg, davies_bouldin
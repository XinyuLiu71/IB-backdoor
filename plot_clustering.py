import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.lines as mlines

def plot_tsne(t_tsne, labels, is_backdoor, epoch, outputs_dir, prefix='t'):
    # 设置配色：使用 muted 颜色
    palette = sns.color_palette("muted", n_colors=10)  # 10个类
    palette.append((1.0, 0.0, 0.0))  # 添加红色用于 backdoor 类
    
    # 创建一个新的标签数组，将 backdoor 数据标记为 10
    combined_labels = labels.copy()
    combined_labels[is_backdoor == 1] = 10  # 将 backdoor 标记为 10

    # 绘图设置
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=t_tsne[:, 0], y=t_tsne[:, 1], 
        hue=combined_labels, 
        palette=palette, 
        s=20,  # 点大小
        # alpha=0.8,  # 透明度
        edgecolor=None  # 去除点边框
    )
    
    # 设置标题和坐标轴字体大小
    # plt.title(f"t-SNE of {prefix} at Epoch {epoch}", fontsize=18)
    # plt.xlabel("t-SNE Component 1", fontsize=16)
    # plt.ylabel("t-SNE Component 2", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 自定义图例
    legend_labels = [f'Class {i}' for i in range(10)] + ['Backdoor']
    custom_lines = [mlines.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=palette[i], markersize=10) for i in range(10)]
    custom_lines.append(mlines.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor='red', markersize=10))

    # 调整图例位置和字体大小
    plt.legend(
        handles=custom_lines, 
        labels=legend_labels, 
        title="Class", 
        loc="upper right", 
        fontsize=12, 
        title_fontsize=14, 
        frameon=True, 
        # shadow=False
    )

    # 保存图像
    save_path = os.path.join(outputs_dir, f'tsne_{prefix}_epoch_{epoch}_improved.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Improved t-SNE plot saved to: {save_path}")

# 输入参数
prefix = 't'
epoch = 120
outputs_dir = 'results/wanet/ob_infoNCE_12_18_0.1_0.5+0.6'

# 加载数据
t_tsne = np.load(os.path.join(outputs_dir, f'tsne_{prefix}_epoch_{epoch}.npy'))
labels = np.load(os.path.join(outputs_dir, f'labels_{prefix}_epoch_{epoch}.npy'))
is_backdoor = np.load(os.path.join(outputs_dir, f'is_backdoor_{prefix}_epoch_{epoch}.npy'))

# 调用绘图函数
plot_tsne(t_tsne, labels, is_backdoor, epoch, outputs_dir, prefix=prefix)

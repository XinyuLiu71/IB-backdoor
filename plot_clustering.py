import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.lines as mlines

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

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
        s=40,  # 调整点大小
        alpha=0.8,  # 设置点透明度
        edgecolor=None,  # 去除点边框
        legend=False  # 去除图例
    )
    
    # 设置坐标轴字体大小
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 自定义图例
    legend_labels = [str(i) for i in range(10)] + ['Backdoor']  # 移除 "Class"
    custom_lines = [mlines.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=palette[i], markersize=20) for i in range(10)]
    custom_lines.append(mlines.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor='red', markersize=20))

    # 调整图例位置为多行多列
    # plt.legend(
    #     handles=custom_lines, 
    #     labels=legend_labels, 
    #     loc="lower center",  # 图例放置在图形下方
    #     # bbox_to_anchor=(0.5, -0.2),  # 图例的位置偏移
    #     ncol=4,  # 设置为 6 列
    #     fontsize=40,  # 图例字体大小
    #     frameon=True, 
    #     fancybox=True,  # 圆角边框
    #     shadow=False,  # 无阴影
    #     columnspacing=0.1,  # 列间距
    #     handletextpad=0.1,  # 线条与文字的间距
    #     labelspacing=0.1  # 图例项之间的间距
    # )

    # 保存图像
    save_path = os.path.join(outputs_dir, f'tsne_{prefix}_epoch_{epoch}_improved.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Adjusted t-SNE plot saved to: {save_path}")

# 输入参数
prefix = 't'
epoch = 40
outputs_dir = 'results/cifar10/adaptive_blend/ob_infoNCE_13_29_0.1_0.4+0.4'

# 加载数据
t_tsne = np.load(os.path.join(outputs_dir, f'tsne_{prefix}_epoch_{epoch}.npy'))
labels = np.load(os.path.join(outputs_dir, f'labels_{prefix}_epoch_{epoch}.npy'))
is_backdoor = np.load(os.path.join(outputs_dir, f'is_backdoor_{prefix}_epoch_{epoch}.npy'))

# 调用绘图函数
plot_tsne(t_tsne, labels, is_backdoor, epoch, outputs_dir, prefix=prefix)

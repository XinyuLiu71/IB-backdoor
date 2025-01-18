import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

def generate_epochs_from_files(directory):
    epochs = []
    pattern = re.compile(r'mi_plot_outputs-vs-Y_epoch_(\d+)\.png')
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            epochs.append(epoch)
    return sorted(epochs)

def plot_individual_mi(MI_dict, title, save_path, epochs, show_legend=True):
    """
    绘制单独的 MI 图像
    """
    # 创建单独的图
    plt.figure(figsize=(8, 6))
    
    # 设置子图背景色和网格线颜色
    ax = plt.gca()  # 获取当前坐标轴
    ax.set_facecolor('#f2f2f2')  # 浅灰色背景
    ax.grid(color='white', linestyle='-', linewidth=6, alpha=0.8)  # 白色网格线

    # 推荐的柔和颜色
    custom_colors = {
        "0_clean": "#2ca02c",  # 柔和的绿色
        "0_backdoor": "#d62728",  # 柔和的红色
        "0_sample": "#ff7f0e",   # 柔和的橙色
        "0": "#1f77b4",  # 柔和的蓝色
        "1": "#9467bd",  # 柔和的紫色
        "2": "#8c564b",  # 柔和的棕色
    }

    # 绘制曲线
    for idx, (class_idx, mi_values) in enumerate(MI_dict.items()):
        # 跳过 Class 1
        if class_idx == 3:
            continue

        # 重命名 Class 2 和 Class 3 为 1 和 2
        if class_idx == 1:
            label_override = "1"
        elif class_idx == 2:
            label_override = "2"
        else:
            label_override = None

        # 设置样式
        label, linestyle, marker, color = set_plot_style(class_idx, idx, custom_colors, label_override)
        mi_estimates = [np.mean(epoch_mi[-5:]) for epoch_mi in mi_values if len(epoch_mi) >= 5]

        # 控制标记间隔
        min_length = min(len(epochs), len(mi_estimates))
        plt.plot(epochs[:min_length], mi_estimates[:min_length],
                 label=label, linestyle=linestyle, linewidth=5 if label in ['0 Backdoor', '0 Sample'] else 3,
                 color=color, marker=marker, markersize=13 if label == '0 Backdoor' else 11, alpha=0.85)

    # 添加标题、标签和图例
    plt.xlabel("Epochs", fontsize=40)
    plt.ylabel("Mutual Information", fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=40)
    
    if show_legend:
        # 获取当前图例的句柄和标签
        handles, labels = plt.gca().get_legend_handles_labels()

        # 手动调整顺序，将宽线条和窄线条分组
        # handles = [handles[1], handles[2], handles[3], handles[0], handles[4], handles[5]]
        # labels = [labels[1], labels[2], labels[3], labels[0], labels[4], labels[5]]
        handles = [handles[1], handles[2], handles[0], handles[3],handles[4]]
        labels = [labels[1], labels[2], labels[0], labels[3], labels[4]]
        plt.legend(handles, labels, loc='lower right', fontsize=33, frameon=True, framealpha=0.7, fancybox=True, shadow=False, borderaxespad=0.1, ncol=2, columnspacing=0.2)

    # 保存图像为 PNG 和 PDF
    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"{title} plot saved to {save_path}.png and {save_path}.pdf")

def set_plot_style(class_idx, idx, custom_colors, label_override=None):
    # 设置样式：重点是区分 `Class 0 Backdoor`, `Class 0`, 以及其他类别
    if str(class_idx) in ["0_backdoor", "0_clean", "0_sample", "0"]:
        if "backdoor" in str(class_idx):
            color, linestyle, marker, label = custom_colors["0_backdoor"], '-', '^', '0 Backdoor'
        elif "clean" in str(class_idx):
            color, linestyle, marker, label = custom_colors["0_clean"], '--', 'o', '0 Clean'
        elif "sample" in str(class_idx):
            color, linestyle, marker, label = custom_colors["0_sample"], '-', 's', '0 Sample'
        else:  # Class 0 overall
            color, linestyle, marker, label = custom_colors["0"], '-', 's', '0'
    else:  # Class 1, 2
        color = custom_colors[str(class_idx)] if str(class_idx) in custom_colors else '#7f7f7f'
        linestyle, marker, label = '--', '', f'{class_idx}'  # 去掉 "Class"
    # 覆盖标签
    if label_override:
        label = label_override
    return label, linestyle, marker, color

def main(args):
    directory = args.directory
    epochs = generate_epochs_from_files(directory)

    MI_inputs_vs_outputs = np.load(f"{directory}/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
    MI_Y_vs_outputs = np.load(f"{directory}/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()

    # 绘制 I(X;T) 图像并保存（不显示图例）
    plot_individual_mi(MI_inputs_vs_outputs, "I(X;T)", os.path.join(directory, 'I_XT_plot'), epochs, show_legend=False)

    # 绘制 I(T;Y) 图像并保存（显示图例）
    plot_individual_mi(MI_Y_vs_outputs, "I(T;Y)", os.path.join(directory, 'I_TY_plot'), epochs, show_legend=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Information Plane")
    parser.add_argument("--directory", type=str, default="results/label_consistent/ob_infoNCE_13_8_0.1_0.4+0.4",
                        help="Directory containing the data files")
    args = parser.parse_args()
    main(args)

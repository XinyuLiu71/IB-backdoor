import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re

def generate_epochs_from_files(directory):
    epochs = []
    pattern = re.compile(r'mi_plot_outputs-vs-Y_epoch_(\d+)\.png')
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            epochs.append(epoch)
    return sorted(epochs)

def plot_combined_mi(MI_dict_XT, MI_dict_YT, args, epochs):
    # 创建子图，1行2列
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    plt.subplots_adjust(wspace=0.25, top=0.85)  # 调整子图间距和上边距

    # 使用 Seaborn 柔和调色板
    colors = sns.color_palette("muted", len(MI_dict_XT))  
    titles = ['I(X;T)', 'I(T;Y)']
    data_dicts = [MI_dict_XT, MI_dict_YT]

    for ax, MI_dict, title in zip(axes, data_dicts, titles):
        for idx, (class_idx, mi_values) in enumerate(MI_dict.items()):
            # 设置样式
            label, linestyle, marker, color = set_plot_style(class_idx, idx, colors)
            mi_estimates = [np.mean(epoch_mi[-5:]) for epoch_mi in mi_values if len(epoch_mi) >= 5]

            # 控制标记间隔
            min_length = min(len(epochs), len(mi_estimates))
            ax.plot(epochs[:min_length], mi_estimates[:min_length],
                    label=label, linestyle=linestyle, linewidth=2.5 if idx < 3 else 1.5,
                    color=color, marker=marker, markersize=8)

        # 子图设置
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("Epochs", fontsize=20)
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_facecolor('white')

    # 设置统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=20, 
               frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.5, 1.1))
    # 设置刻度字体和加粗
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # 保存图像
    output_dir = args.directory
    output_path = os.path.join(output_dir, 'combined_mi_plots_optimized.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimized MI plot saved to {output_path}")

def set_plot_style(class_idx, idx, colors):
    # 重点类使用更醒目的样式
    if class_idx in ["0_backdoor", "0_clean", "0_sample"]:
        if "backdoor" in class_idx:
            color, linestyle, marker, label = 'crimson', '-', '^', 'Class 0 Backdoor'
        elif "clean" in class_idx:
            color, linestyle, marker, label = 'forestgreen', '-', 'o', 'Class 0 Clean'
        elif "sample" in class_idx:
            color, linestyle, marker, label = 'mediumblue', '-', 's', 'Class 0 Sample'
    else:
        color = colors[idx % len(colors)]
        linestyle, marker, label = '--', '', f'Class {class_idx}'
    return label, linestyle, marker, color

def main(args):
    directory = args.directory
    epochs = generate_epochs_from_files(directory)

    MI_inputs_vs_outputs = np.load(f"{directory}/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
    MI_Y_vs_outputs = np.load(f"{directory}/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()

    plot_combined_mi(MI_inputs_vs_outputs, MI_Y_vs_outputs, args, epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Information Plane")
    parser.add_argument("--directory", type=str, default="results/badnet/ob_infoNCE_11_32_0.1_0.6+0.6",
                        help="Directory containing the data files")
    args = parser.parse_args()
    main(args)

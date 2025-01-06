import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import argparse
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
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

def plot_metric(metric_dict, type, args, epochs):
    plt.figure(figsize=(12, 8))
    # colors = plt.cm.get_cmap('tab10', 10)  # 使用颜色映射
    colors = list(mcolors.TABLEAU_COLORS.values())

    for class_idx, metric_values in metric_dict.items():
        if class_idx in ['0_backdoor', '0_clean', '0_sample']:
            # 对于backdoor和clean样本，使用虚线
            label = f'Class 0 {"Backdoor" if "backdoor" in class_idx else "Clean" if "clean" in class_idx else "Sample"}'
            linestyle = '-'
            color = 'red' if 'backdoor' in class_idx else 'green' if 'clean' in class_idx else 'blue'
            marker = '^' if 'backdoor' in class_idx else 'o' if 'clean' in class_idx else 's'
        else:
            label = f'Class {class_idx}'
            linestyle = '--'
            # color = colors(class_idx)
            color = colors[class_idx % len(colors)]
            if class_idx == 2:
                color = colors[4]
            marker = 's' if class_idx == 0 else None

        # 如果有些 epoch 的 MI 估计次数少于 5 次，我们可以选择跳过这些 epoch 或使用所有可用的估计值
        # mi_estimates = [np.mean(epoch_mi[-5:]) if len(epoch_mi) >= 5 else np.mean(epoch_mi) for epoch_mi in mi_values]
        print(f"Class {class_idx}:")
        print(f"  Number of epochs: {len(epochs)}")
        # 确保 epochs 和 mi_estimates 长度匹配
        min_length = min(len(epochs), len(metric_values))
        epochs_to_plot = epochs[:min_length]
        mi_estimates_to_plot = metric_values[:min_length]
        
        plt.plot(epochs_to_plot, mi_estimates_to_plot, label=label, linestyle=linestyle, linewidth=2, color=color, marker=marker, markersize=10)

    # 设置字体大小
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Silhouette Scores', fontsize=10)
    plt.title(f'Silhouette Scores of {type} over Training Epochs', fontsize=10)
    plt.legend(fontsize=10)

    # 设置刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 设置背景阴影颜色
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')  # 浅灰色背景
    ax.patch.set_alpha(0.9)      # 背景透明度

    # 设置网格线为白色
    plt.grid(True, color='white', linestyle='-', linewidth=1.2, alpha=1)

    plt.tight_layout()

    # 保存图像
    output_dir = args.directory
    plt.savefig(os.path.join(output_dir, f'metric_{type}_plot.png'))
    plt.close()

    print(f"Scores plot for {type} saved to {os.path.join(output_dir, f'scores_{type}_plot.png')}")

def main(args):
    directory = args.directory
    observe_class = args.observe_class

    # epochs = generate_epochs_from_files(directory)
    epochs = [5, 10, 20, 40, 60, 80, 100, 120]
    print(f"epochs: {epochs}")
    metrics_t = {}
    metrics_pred = {}

    t_arr = np.load(os.path.join(directory, 'metrics_t.npy'), allow_pickle=True).item()
    # pred_arr = np.load(os.path.join(directory, 'metrics_pred.npy'), allow_pickle=True).item()

    # Assuming MI_dict is available, you would call plot_mi like this:
    plot_metric(t_arr, 'T', args, epochs)
    # plot_metric(pred_arr, 'Y_pred', args, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Information Plane")
    parser.add_argument("--directory", type=str, default="results/blend/ob_infoNCE_12_19_0.1_0.4+0.4", help="Directory containing the data files")
    parser.add_argument("--observe_class", type=list, default=[0,'0_backdoor', '0_clean', 1,2,3,4,5,6,7,8,9], help="Class to observe")
    args = parser.parse_args()

    main(args)

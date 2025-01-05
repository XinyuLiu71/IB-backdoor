import matplotlib.pyplot as plt
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

def load_data(directory, observe_class):
    inputs_outputs_arr = np.load(f"{directory}/infoNCE_MI_I(X,T)_class_{observe_class}.npy", allow_pickle=True)
    Y_outputs_arr = np.load(f"{directory}/infoNCE_MI_I(Y,T)_class_{observe_class}.npy", allow_pickle=True)
    return inputs_outputs_arr, Y_outputs_arr

def process_data(inputs_outputs_arr, Y_outputs_arr, epochs):
    info_plane = np.empty([len(epochs), 2])
    for idx in range(len(epochs)):
        info_plane[idx, 0] = np.mean(inputs_outputs_arr[idx][-5:])
        info_plane[idx, 1] = np.mean(Y_outputs_arr[idx][-5:])
    return info_plane

def create_dataframe(epochs, info_plane):
    df = pd.DataFrame(columns=['Epoch', 'I(X;T)', 'I(T;Y)'])
    df['Epoch'] = epochs
    df['I(X;T)'] = info_plane[:, 0]
    df['I(T;Y)'] = info_plane[:, 1]
    return df

def plot_info_plane(df, observe_class, save_path):
    fig, ax = plt.subplots()
    sca = ax.scatter(x=df['I(X;T)'], y=df['I(T;Y)'], c=df['Epoch'], cmap='summer')
    ax.set_xlabel('I(X;T)')
    ax.set_ylabel('I(T;Y)')
    fig.colorbar(sca, label="Epoch", orientation="vertical")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {save_path}")

def plot_mi(MI_dict, mi_type, args, epochs):
    plt.figure(figsize=(12, 8))
    
    for class_idx, mi_values in MI_dict.items():
        # 计算每个 epoch 的 MI 估计值（最后 5 次的平均值）
        mi_estimates = [np.mean(epoch_mi[-5:]) for epoch_mi in mi_values if len(epoch_mi) >= 5]
        
        # 如果有些 epoch 的 MI 估计次数少于 5 次，我们可以选择跳过这些 epoch 或使用所有可用的估计值
        # mi_estimates = [np.mean(epoch_mi[-5:]) if len(epoch_mi) >= 5 else np.mean(epoch_mi) for epoch_mi in mi_values]
        print(f"Class {class_idx}:")
        print(f"  Number of epochs: {len(epochs)}")
        print(f"  Number of MI estimates: {len(mi_estimates)}")
        # 确保 epochs 和 mi_estimates 长度匹配
        min_length = min(len(epochs), len(mi_estimates))
        epochs_to_plot = epochs[:min_length]
        mi_estimates_to_plot = mi_estimates[:min_length]
        
        # epochs = range(1, len(mi_estimates) + 1)
        # plt.plot(epochs, mi_estimates, label=f'Class {class_idx}')
        plt.plot(epochs_to_plot, mi_estimates_to_plot, label=f'Class {class_idx}')

    
    plt.xlabel('Epochs')
    plt.ylabel('Mutual Information Estimate')
    plt.title(f'Mutual Information ({mi_type}) over Training Epochs')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    output_dir = args.directory
    plt.savefig(os.path.join(output_dir, f'mi_{mi_type}_plot.png'))
    plt.close()
    
    print(f"MI plot for {mi_type} saved to {os.path.join(output_dir, f'mi_{mi_type}_plot.png')}")

def main(args):
    directory = args.directory
    observe_class = args.observe_class

    epochs = generate_epochs_from_files(directory)
    print(f"epochs: {epochs}")
    MI_inputs_vs_outputs = {}
    MI_Y_vs_outputs = {}
    for cls in observe_class:
        inputs_outputs_arr, Y_outputs_arr = load_data(directory, cls)
        
        print("Shape of inputs_outputs_arr:", inputs_outputs_arr.shape)
        print("Shape of Y_outputs_arr:", Y_outputs_arr.shape)

        info_plane = process_data(inputs_outputs_arr, Y_outputs_arr, epochs)
        df = create_dataframe(epochs, info_plane)

        save_path = os.path.join(directory, f"info_plane_class_{cls}.png")
        plot_info_plane(df, cls, save_path)

        MI_inputs_vs_outputs[cls] = inputs_outputs_arr
        MI_Y_vs_outputs[cls] = Y_outputs_arr

    # Assuming MI_dict is available, you would call plot_mi like this:
    plot_mi(MI_inputs_vs_outputs, 'inputs_vs_outputs', args, epochs)
    plot_mi(MI_Y_vs_outputs, 'Y_vs_outputs', args, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Information Plane")
    parser.add_argument("--directory", type=str, default="results/badnet/ob_infoNCE_10_25_0.1_test", help="Directory containing the data files")
    parser.add_argument("--observe_class", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="Class to observe")
    args = parser.parse_args()

    main(args)

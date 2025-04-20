import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import re

def generate_epochs_from_files(directory):
    epochs = []
    # pattern = re.compile(r'mi_plot_inputs-vs-outputs_epoch_(\d+)\.png')
    # pattern = re.compile(r'mi_plot_inputs-vs-Y_epoch_(\d+)\.png')
    pattern = re.compile(r'IXY_class_all_epoch_(\d+)\.png')
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            epochs.append(epoch)
    
    return sorted(epochs)

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
        if class_idx in ['0_sample', '0_backdoor', '0_clean']:
            # 对于backdoor和clean样本，使用虚线
            label = f'Class 0 {"Backdoor" if "backdoor" in class_idx else "Clean" if "clean" in class_idx else "Sample"}'
            linestyle = '--'
        else:
            label = f'Class {class_idx}'
            linestyle = '-'
        
        # 计算每个 epoch 的 MI 估计值（最后 5 次的平均值）
        mi_estimates = [np.mean(epoch_mi[-5:]) for epoch_mi in mi_values if len(epoch_mi) >= 5]
        
        # 如果有些 epoch 的 MI 估计次数少于 5 次，我们可以选择跳过这些 epoch 或使用所有可用的估计值
        # mi_estimates = [np.mean(epoch_mi[-5:]) if len(epoch_mi) >= 5 else np.mean(epoch_mi) for epoch_mi in mi_values]
        print(f"{label}:")
        print(f"  Number of epochs: {len(epochs)}")
        print(f"  Number of MI estimates: {len(mi_estimates)}")
        # 确保 epochs 和 mi_estimates 长度匹配
        min_length = min(len(epochs), len(mi_estimates))
        epochs_to_plot = epochs[:min_length]
        mi_estimates_to_plot = mi_estimates[:min_length]
        
        plt.plot(epochs_to_plot, mi_estimates_to_plot, label=label, linestyle=linestyle)

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

def plot_mi_all(MI_inputs_vs_outputs, MI_Y_vs_outputs, MI_inputs_vs_Y, args, epochs):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(12, 8))

    # 设置子图背景色和网格线颜色
    ax = plt.gca()  # 获取当前坐标轴
    ax.set_facecolor('#f2f2f2')  # 浅灰色背景
    ax.grid(color='white', linestyle='-', linewidth=6, alpha=0.8)  # 白色网格线

    # Plot I(X;T)
    mi_estimates_xt = [np.mean(epoch_mi[-5:]) for epoch_mi in MI_inputs_vs_outputs['all'] if len(epoch_mi) >= 5]
    min_length = min(len(epochs), len(mi_estimates_xt))
    epochs_to_plot = epochs[:min_length]
    mi_estimates_xt_to_plot = mi_estimates_xt[:min_length]
    plt.plot(epochs_to_plot, mi_estimates_xt_to_plot, label=r'$I(X;T)$', linestyle='-', linewidth=5)

    # Plot I(T;Y)
    mi_estimates_ty = [np.mean(epoch_mi[-5:]) for epoch_mi in MI_Y_vs_outputs['all'] if len(epoch_mi) >= 5]
    min_length = min(len(epochs), len(mi_estimates_ty))
    epochs_to_plot = epochs[:min_length]
    mi_estimates_ty_to_plot = mi_estimates_ty[:min_length]
    plt.plot(epochs_to_plot, mi_estimates_ty_to_plot, label=r'$I(T;Y_{pred})$', linestyle='-', linewidth=5)

    # Plot I(X;Y)
    mi_estimates_xy = [np.mean(epoch_mi[-5:]) for epoch_mi in MI_inputs_vs_Y['all'] if len(epoch_mi) >= 5]
    min_length = min(len(epochs), len(mi_estimates_xy))
    epochs_to_plot = epochs[:min_length]
    mi_estimates_xy_to_plot = mi_estimates_xy[:min_length]
    plt.plot(epochs_to_plot, mi_estimates_xy_to_plot, label=r'$I(X;Y_{pred})$', linestyle='-', linewidth=5)
    
    plt.xlabel('Epochs', fontsize=40)
    plt.ylabel('Mutual Information Estimate', fontsize=40)
    plt.title('Mutual Information over Training Epochs', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.legend(loc='upper right', fontsize=35, frameon=True, framealpha=0.7, fancybox=True, shadow=False, bbox_to_anchor=(0.95, 0.95))
    plt.grid(True)
    plt.tight_layout()  

    # Save the plot
    output_dir = args.directory
    plt.savefig(os.path.join(output_dir, 'mi_all_curves_plot.pdf'))
    plt.close()
    
    print(f"MI plots saved to {output_dir}/mi_all_curves_plot.pdf")

def main(args):
    directory = args.directory
    observe_class = args.observe_class

    epochs = generate_epochs_from_files(directory)
    print(f"epochs: {epochs}")

    MI_inputs_vs_outputs = np.load(f"{directory}/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
    MI_Y_vs_outputs = np.load(f"{directory}/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()
    MI_inputs_vs_Y = np.load(f"{directory}/infoNCE_MI_I(X,Y).npy", allow_pickle=True).item()

    # Plot all MI curves on the same figure
    plot_mi_all(MI_inputs_vs_outputs, MI_Y_vs_outputs, MI_inputs_vs_Y, args, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Information Plane")
    parser.add_argument("--directory", type=str, default="results/cifar10/blend/ob_infoNCE_14_08all_0.1_0.4+0.4", help="Directory containing the data files")
    parser.add_argument("--observe_class", type=list, default=['all'], help="Class to observe")
    args = parser.parse_args()

    main(args)
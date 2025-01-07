import numpy as np
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

def calculate_normalized_mi_difference(MI_dict, target_class, other_classes, selected_epochs_index):
    """
    计算目标类别和其他类别的 MI 差值（按每个 epoch 单独归一化）。
    
    参数：
    - MI_dict: 包含所有类别 MI 值的字典
    - target_class: 目标类别的键（如 0 或 '0_clean'）
    - other_classes: 其他类别的键列表
    - selected_epochs_index: 选定的 epoch 索引列表
    
    返回：
    - 平均差值
    """
    # 初始化存储差值的列表
    differences = []

    # 对每个 epoch 分别计算 MI 差值
    for epoch in selected_epochs_index:
        # 提取目标类别在当前 epoch 的 MI
        target_mi_epoch = np.mean(MI_dict[target_class][epoch][-5:])

        # 提取其他类别在当前 epoch 的 MI
        other_mis_epoch = [
            np.mean(MI_dict[cls][epoch][-5:])
            for cls in other_classes
        ]

        # 归一化处理（基于当前 epoch 的最大值和最小值）
        combined_mis = [target_mi_epoch] + other_mis_epoch
        max_mi = np.max(combined_mis)
        min_mi = np.min(combined_mis)
        # print(f"max_mi: {max_mi}, min_mi: {min_mi}")
        # print(f"before target_mi_epoch: {target_mi_epoch} other_mis_epoch: {other_mis_epoch}")
        target_mi_epoch = (target_mi_epoch - min_mi) / (max_mi - min_mi + 1e-8)
        other_mis_epoch = [
            (mi - min_mi) / (max_mi - min_mi + 1e-8)
            for mi in other_mis_epoch
        ]
        # print(f"after target_mi_epoch: {target_mi_epoch} other_mis_epoch: {other_mis_epoch}")
        # 计算目标类别与每个其他类别的归一化 MI 差值
        differences_epoch = target_mi_epoch - np.array(other_mis_epoch)
        differences.append(np.abs(np.mean(differences_epoch)))  # 保存每个 epoch 的平均差值

    # 对所有 epoch 的差值求平均
    avg_difference = np.mean(differences)

    return avg_difference


if __name__ == "__main__":
    # 设置路径
    dir = "results/blend/ob_infoNCE_11_32_0.1_0.4+0.4"
    epochs = generate_epochs_from_files(dir)

    # 加载 MI 数据
    MI_inputs_vs_outputs = np.load(f"{dir}/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
    MI_Y_vs_outputs = np.load(f"{dir}/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()

    dir2 = "results/blend/ob_infoNCE_12_01_0.1_0.4+0.4"
    MI_inputs_vs_outputs2 = np.load(f"{dir2}/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
    MI_Y_vs_outputs2 = np.load(f"{dir2}/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()

    MI_inputs_vs_outputs = {**MI_inputs_vs_outputs, **MI_inputs_vs_outputs2}
    MI_Y_vs_outputs = {**MI_Y_vs_outputs, **MI_Y_vs_outputs2}

    # 定义目标类别和其他干净类别
    target_clean = "0_clean"  # 类 0 clean 数据
    target_sample = 0  # 类 0 数据
    other_classes = [cls for cls in MI_inputs_vs_outputs.keys() if isinstance(cls, int) and cls != 0]  # 干净类别

    # 选择的 epoch
    selected_epochs = [5, 40, 100]  # 第1、40和最后一个 epoch
    selected_epochs_index = [epochs.index(epoch) for epoch in selected_epochs]

    # 计算类 0 clean 和其他干净类别的 I(X;T) 差值平均
    avg_diff_xt_clean = calculate_normalized_mi_difference(MI_inputs_vs_outputs, target_clean, other_classes, selected_epochs_index)
    print(f"I(X;T) 类 0 clean 和其他类别的 MI 差值平均 (normalized): {avg_diff_xt_clean:.4f}")

    # 计算类 0 sample 和其他干净类别的 MI 差值平均
    avg_diff_xt_sample = calculate_normalized_mi_difference(MI_inputs_vs_outputs, target_sample, other_classes, selected_epochs_index)
    print(f"I(X;T) 类 0 和其他类别的 MI 差值平均 (normalized): {avg_diff_xt_sample:.4f}")

    # 计算类 0 clean 和其他干净类别的I(T;Y) 差值平均
    avg_diff_ty_clean = calculate_normalized_mi_difference(MI_Y_vs_outputs, target_clean, other_classes, selected_epochs_index)
    print(f"I(T;Y) 类 0 clean 和其他类别的 MI 差值平均 (normalized): {avg_diff_ty_clean:.4f}")

    # 计算类 0 sample 和其他干净类别的 MI 差值平均
    avg_diff_ty_sample = calculate_normalized_mi_difference(MI_Y_vs_outputs, target_sample, other_classes, selected_epochs_index)
    print(f"I(T;Y) 类 0 和其他类别的 MI 差值平均 (normalized): {avg_diff_ty_sample:.4f}")

    # 总体 MI 差值
    avg_diff_total_xt = np.abs(avg_diff_xt_clean) + np.abs(avg_diff_xt_sample)
    print(f"I(X;T) 总体差值：{avg_diff_total_xt:.4f}")
    avg_diff_total_ty = np.abs(avg_diff_ty_clean) + np.abs(avg_diff_ty_sample)
    print(f"I(T;Y) 总体差值：{avg_diff_total_ty:.4f}")
    avg_diff_total = np.abs(avg_diff_xt_clean) + np.abs(avg_diff_xt_sample) + np.abs(avg_diff_ty_clean) + np.abs(avg_diff_ty_sample)
    print(f"总体 MI 差值 (normalized): {avg_diff_total:.4f}")

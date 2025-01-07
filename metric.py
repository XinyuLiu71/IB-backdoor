import numpy as np
import os

# 定义计算 Stealth Score 的函数
def calculate_stealth_score(silhouette):
    """
    根据提供的 silhouette 数据计算 Stealth Score，仅使用最后一个 epoch 的数据。
    
    silhouette: 一个字典，包含以下键：
        - 0: 类别 0 的整体轮廓分数
        - '0_clean': 类别 0 中 clean data 的轮廓分数
        - '0_backdoor': 类别 0 中 backdoor data 的轮廓分数
        - 1, 2, ..., 9: 其他类别的轮廓分数
        
    返回：
        - 一个包含 M1, M2 和 Stealth Score 的字典。
    """
    # 取最后一个 epoch 的轮廓分数
    silhouette_all_classes = np.array([silhouette[i][-1] for i in range(1, 10)])
    silhouette_class_0_clean = silhouette['0_clean'][-1]
    silhouette_class_0_backdoor = silhouette['0_backdoor'][-1]

    # M1: 类 0 clean data 的轮廓分数与其他类别的平均轮廓分数的差值绝对值
    M1 = np.abs(silhouette_class_0_clean - np.mean(silhouette_all_classes))

    # 计算类 0 clean 和 backdoor 的样本比例
    D_clean = len(silhouette['0_clean'])
    D_backdoor = len(silhouette['0_backdoor'])
    D_total = D_clean + D_backdoor

    # M2: 结合 clean 和 backdoor 的比例计算
    M2 = (D_clean / D_total) * silhouette_class_0_clean + (D_backdoor / D_total) * silhouette_class_0_backdoor

    # 返回未归一化的 M1 和 M2
    return M1, M2


def normalize_and_compute_stealth_score(M1, M2):
    """
    归一化 M1 和 M2 并计算最终 Stealth Score。
    
    M1: scalar, 计算得到的 M1 值。
    M2: scalar, 计算得到的 M2 值。
    
    返回：
        - Stealth Score（归一化后）。
    """
    # 归一化处理
    M1_normalized = (M1 - np.min([M1, M2])) / (np.max([M1, M2]) - np.min([M1, M2]))
    M2_normalized = (M2 - np.min([M1, M2])) / (np.max([M1, M2]) - np.min([M1, M2]))

    # 设置权重并计算 Stealth Score
    # w1, w2 = 0.5, 0.5
    # stealth_score = w1 * M1_normalized + w2 * M2_normalized
    stealth_score = M1_normalized + M2_normalized
    return stealth_score


# 主程序入口
if __name__ == "__main__":
    # 文件路径
    dir = "results/badnet/ob_infoNCE_12_18_0.1_0.6+0.6"
    file_path = os.path.join(dir, "metrics_t.npy")

    # 加载 silhouette 数据
    silhouette = np.load(file_path, allow_pickle=True).item()

    # 计算 M1 和 M2（仅使用最后一个 epoch 的数据）
    M1, M2 = calculate_stealth_score(silhouette)

    # 计算最终 Stealth Score
    stealth_score = normalize_and_compute_stealth_score(M1, M2)

    # 打印结果
    print(f"M1 (最后一个 epoch): {M1}")
    print(f"M2 (最后一个 epoch): {M2}")
    print(f"Stealth Score (最后一个 epoch): {stealth_score}")

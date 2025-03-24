import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
from openTSNE import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from mpl_toolkits.mplot3d import Axes3D
import umap


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

from mpl_toolkits.mplot3d import Axes3D

def plot_tsne_3d(t_tsne, labels, is_backdoor, epoch, outputs_dir, prefix='t'):
    """
    绘制 t-SNE 的三维可视化。
    参数:
    - t_tsne: t-SNE 降维后的表示 (n_samples, 3)
    - labels: 样本的类别标签
    - is_backdoor: 是否是 backdoor 数据的标记
    - epoch: 当前 epoch
    - outputs_dir: 图像保存路径
    - prefix: 图像文件名前缀
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 将 backdoor 数据标记为一个额外的类别
    combined_labels = labels.copy()
    combined_labels[is_backdoor == 1] = 10

    # 定义颜色映射：0-9 类别 + backdoor 类别
    palette = sns.color_palette("tab10", n_colors=10)
    palette.append('red')  # 为 backdoor 添加红色

    # 绘制 t-SNE 三维散点图
    for i in range(11):  # 0-9 和 backdoor 类
        indices = combined_labels == i
        ax.scatter(t_tsne[indices, 0], t_tsne[indices, 1], t_tsne[indices, 2],
                   color=palette[i], label=f'Class {i}' if i < 10 else 'Backdoor', s=15)

    # 设置图表属性
    ax.set_title(f"3D t-SNE Visualization at Epoch {epoch}", fontsize=14)
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.set_zlabel("t-SNE Component 3", fontsize=12)

    # 图例设置
    ax.legend(loc='upper right', fontsize=10)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f'tsne_3d_{prefix}_epoch_{epoch}.png'), dpi=300)
    plt.close()
    print(f"3D t-SNE plot saved to: tsne_3d_{prefix}_epoch_{epoch}.png")


def plot_tsne(t, labels, is_backdoor, epoch, outputs_dir, prefix='t'):
    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, n_jobs=16)
    # t_tsne = tsne.fit_transform(t.cpu().numpy())
    t_tsne = tsne.fit(t.cpu().numpy())

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


def compute_single_class_metrics(t_tsne, labels, class_id):
    """
    计算单个类别的轮廓系数均值和类内紧凑性/类间分离度。

    参数:
    - t_tsne: t-SNE 降维后的表示 (n_samples, 2)
    - labels: 所有样本的类别标签
    - class_id: 目标类别的ID

    返回:
    - silhouette_mean: 目标类别的平均轮廓系数
    - compactness: 类内紧凑性
    - separation: 类间分离度
    """
    # Step 1: 计算全局的轮廓系数
    silhouette_vals = silhouette_samples(t_tsne, labels)

    # 提取目标类别的轮廓系数
    class_indices = np.where(labels == class_id)[0]
    silhouette_mean = np.mean(silhouette_vals[class_indices])
    
    # Step 2: 计算类内紧凑性
    # class_points = t_tsne[class_indices]
    # center_c = np.mean(class_points, axis=0)
    # compactness = np.mean(np.linalg.norm(class_points - center_c, axis=1))
    
    # # Step 3: 计算类间分离度
    # unique_classes = np.unique(labels)
    # separations = []
    # for other_class in unique_classes:
    #     if other_class != class_id:
    #         other_indices = np.where(labels == other_class)[0]
    #         center_j = np.mean(t_tsne[other_indices], axis=0)
    #         separation = np.linalg.norm(center_c - center_j)
    #         separations.append(separation)
    # separation_min = min(separations)  # 选择到其他簇的最小距离
    
    # 返回结果
    # return silhouette_mean, compactness/separation_min
    return silhouette_mean



# def analyze_and_visualize(data, labels, is_backdoor, epoch, outputs_dir, prefix='t'):
#     """
#     主函数，执行 t-SNE 降维、计算聚类指标并绘制可视化图。
    
#     参数:
#     - t: 原始特征表示 (Tensor)
#     - labels: 样本的类别标签 (Tensor)
#     - is_backdoor: 是否是 backdoor 数据的标记 (Tensor)
#     - epoch: 当前 epoch
#     - outputs_dir: 图像保存路径
#     - prefix: 图像文件名前缀
#     """
#     # 转换数据为 numpy 格式
#     labels_np = labels.cpu().numpy()
#     is_backdoor_np = is_backdoor.cpu().numpy()

#     # 使用 t-SNE 降维
#     tsne = TSNE(n_components=2, random_state=42, n_jobs=16)
#     # t_tsne = tsne.fit_transform(data.cpu().numpy())
#     t_tsne = tsne.fit(data.cpu().numpy())

#     if epoch == 60 or epoch == 120:
#         np.save(os.path.join(outputs_dir, f'tsne_{prefix}_epoch_{epoch}.npy'), t_tsne)
#         np.save(os.path.join(outputs_dir, f'labels_{prefix}_epoch_{epoch}.npy'), labels_np)
#         np.save(os.path.join(outputs_dir, f'is_backdoor_{prefix}_epoch_{epoch}.npy'), is_backdoor_np)

#     # 计算每个类别的指标
#     metrics = {}
#     # Step 1: 计算全局的轮廓系数
#     silhouette_vals = silhouette_samples(t_tsne, labels_np)

#     print("### Per-Class Clustering Metrics ###")
#     for class_id in range(10):
#         # 提取目标类别的轮廓系数
#         class_indices = np.where(labels_np == class_id)[0]
#         silhouette_mean = np.mean(silhouette_vals[class_indices])
#         metrics[class_id] = silhouette_mean
#         print(f"Class {class_id} {prefix} silhouette: {silhouette_mean:.4f}")

#         # class_points = t_tsne[class_indices]
#         # center_c = np.mean(class_points, axis=0)
#         # compactness = np.mean(np.linalg.norm(class_points - center_c, axis=1))
#         # metrics[class_id] = compactness
#         # print(f"Class {class_id} {prefix} compactness: {compactness:.4f}")

#     # 计算类别 0 中 clean 和 backdoor 数据的指标
#     print("### Class 0 Subgroup Metrics ###")
#     # 将 backdoor 和 clean 作为独立类别
#     extended_labels = labels_np.copy()
#     extended_labels[(labels_np == 0) & (is_backdoor_np == 0)] = 10  # clean -> 类别 10
#     extended_labels[(labels_np == 0) & (is_backdoor_np == 1)] = 11  # backdoor -> 类别 11
#     silhouette_vals = silhouette_samples(t_tsne, extended_labels)

#     class0_clean_indices = np.where(extended_labels==10)[0]
#     class0_backdoor_indices = np.where(extended_labels==11)[0]

#     silhouette_clean = np.mean(silhouette_vals[class0_clean_indices])
#     silhouette_backdoor = np.mean(silhouette_vals[class0_backdoor_indices])

#     metrics['0_clean'] = silhouette_clean
#     metrics['0_backdoor'] = silhouette_backdoor
    
#     print(f"Class 0 Clean {prefix} silhouette: {silhouette_clean:.4f}")
#     print(f"Class 0 Backdoor {prefix} silhouette: {silhouette_backdoor:.4f}")
#     # class0_clean_points = t_tsne[class0_clean_indices]
#     # class0_backdoor_points = t_tsne[class0_backdoor_indices]
#     # center_clean = np.mean(class0_clean_points, axis=0)
#     # center_backdoor = np.mean(class0_backdoor_points, axis=0)
#     # compactness_clean = np.mean(np.linalg.norm(class0_clean_points - center_clean, axis=1))
#     # compactness_backdoor = np.mean(np.linalg.norm(class0_backdoor_points - center_backdoor, axis=1))
#     # metrics['0_clean'] = compactness_clean
#     # metrics['0_backdoor'] = compactness_backdoor
#     # print(f"Class 0 Clean {prefix} compactness: {compactness_clean:.4f}")
#     # print(f"Class 0 Backdoor {prefix} compactness: {compactness_backdoor:.4f}")

#     # 绘制 t-SNE 图
#     plot_tsne(t_tsne, labels_np, is_backdoor_np, epoch, outputs_dir, prefix)
#     # plot_tsne_3d(t_tsne, labels_np, is_backdoor_np, epoch, outputs_dir, prefix)
#     return metrics

def plot_umap(t_umap, labels, is_backdoor, epoch, outputs_dir, prefix='t'):
    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))

    # 创建一个新的标签数组，将类别标签与是否是 backdoor 数据的标记结合
    # 将 backdoor 数据标记为 10（一个额外的类别），其他类别保留原标签
    
    combined_labels = labels.copy()
    combined_labels[is_backdoor == 1] = 10
    
    # 创建一个颜色映射，0-9 是类别，10 是 backdoor
    palette = sns.color_palette("tab10", n_colors=10)  # 使用 10 个颜色表示 10 个类别
    palette.append('red')  # 添加一个颜色来表示 backdoor 类
    
    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=t_umap[:, 0], y=t_umap[:, 1], hue=combined_labels, 
                    palette=palette, legend='full', marker='o')
    
    # 设置标题和轴标签
    plt.title(f'UMAP of {prefix} at Epoch {epoch}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')

    # 自定义图例，确保 'backdoor' 的颜色是红色
    # 创建图例条目
    legend_labels = [f'Class {i}' for i in range(10)] + ['Backdoor']
    custom_lines = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in range(10)]
    # 添加 'Backdoor' 红色图例
    custom_lines.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10))

    # 添加图例
    plt.legend(handles=custom_lines, labels=legend_labels, title='Class')

    # 保存图像
    plt.savefig(os.path.join(outputs_dir, f'umap_{prefix}_epoch_{epoch}.png'))
    plt.close()

def plot_umap_3d(t_umap, labels, is_backdoor, epoch, outputs_dir, prefix='t'):
    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))

    # 创建一个新的标签数组，将类别标签与是否是 backdoor 数据的标记结合
    # 将 backdoor 数据标记为 10（一个额外的类别），其他类别保留原标签
    
    combined_labels = labels.copy()
    combined_labels[is_backdoor == 1] = 10
    
    # 创建一个颜色映射，0-9 是类别，10 是 backdoor
    palette = sns.color_palette("tab10", n_colors=10)  # 使用 10 个颜色表示 10 个类别
    palette.append('red')  # 添加一个颜色来表示 backdoor 类
    
    # 3D 可视化降维结果
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        t_umap[:, 0], t_umap[:, 1], t_umap[:, 2], 
        c=labels, cmap='tab10', s=10, alpha=0.7
    )
    ax.set_title(f"3D UMAP Visualization - Epoch {epoch}")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    fig.colorbar(scatter, label="Class Label")
    plt.savefig(os.path.join(outputs_dir, f"{prefix}_umap_epoch_{epoch}_3d.png"))
    plt.close()

def batch_umap(data, reducer, batch_size=1000):
    """
    使用分批处理对大规模数据进行降维。
    
    参数:
    - data: 大规模数据集 (numpy array)
    - reducer: UMAP 降维对象
    - batch_size: 每批数据的大小
    
    返回:
    - data_umap: 所有数据降维后的结果
    """
    num_samples = data.shape[0]
    data_umap = []
    
    for i in range(0, num_samples, batch_size):
        batch = data[i:i+batch_size]
        batch_umap = reducer.fit_transform(batch)
        data_umap.append(batch_umap)
    
    return np.vstack(data_umap)

def analyze_and_visualize_umap(data, labels, is_backdoor, epoch, outputs_dir, prefix='u'):
    """
    主函数，执行 UMAP 降维、计算聚类指标并绘制可视化图。
    
    参数:
    - data: 原始特征表示 (Tensor)
    - labels: 样本的类别标签 (Tensor)
    - is_backdoor: 是否是 backdoor 数据的标记 (Tensor)
    - epoch: 当前 epoch
    - outputs_dir: 图像保存路径
    - prefix: 图像文件名前缀
    """
    # 转换数据为 numpy 格式
    labels_np = labels.cpu().numpy()
    is_backdoor_np = is_backdoor.cpu().numpy()

    # 使用 UMAP 降维
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    # data_np = data.cpu().numpy()
    data_np = np.ascontiguousarray(data.cpu().numpy().astype(np.float32))
    # data_umap = reducer.fit_transform(data_np)
    data_umap = batch_umap(data_np, reducer, batch_size=1000)

    plot_umap(data_umap, labels_np, is_backdoor_np, epoch, outputs_dir, prefix)


    # 保存降维结果（仅在特定 epoch 保存）
    if epoch in [5, 60, 120]:
        np.save(os.path.join(outputs_dir, f'umap_{prefix}_epoch_{epoch}.npy'), data_umap)
        np.save(os.path.join(outputs_dir, f'labels_{prefix}_epoch_{epoch}.npy'), labels_np)
        np.save(os.path.join(outputs_dir, f'is_backdoor_{prefix}_epoch_{epoch}.npy'), is_backdoor_np)

    # 计算每个类别的指标
    metrics = {}
    silhouette_vals = silhouette_samples(data_umap, labels_np)

    print("### Per-Class Clustering Metrics ###")
    for class_id in range(10):
        # 提取目标类别的轮廓系数
        class_indices = np.where(labels_np == class_id)[0]
        silhouette_mean = np.mean(silhouette_vals[class_indices])
        metrics[class_id] = silhouette_mean
        print(f"Class {class_id} {prefix} silhouette: {silhouette_mean:.4f}")

    # 计算类别 0 中 clean 和 backdoor 数据的指标
    print("### Class 0 Subgroup Metrics ###")
    # 将 backdoor 和 clean 作为独立类别
    extended_labels = labels_np.copy()
    extended_labels[(labels_np == 0) & (is_backdoor_np == 0)] = 10  # clean -> 类别 10
    extended_labels[(labels_np == 0) & (is_backdoor_np == 1)] = 11  # backdoor -> 类别 11
    silhouette_vals = silhouette_samples(data_umap, extended_labels)

    class0_clean_indices = np.where(extended_labels == 10)[0]
    class0_backdoor_indices = np.where(extended_labels == 11)[0]

    silhouette_clean = np.mean(silhouette_vals[class0_clean_indices])
    silhouette_backdoor = np.mean(silhouette_vals[class0_backdoor_indices])

    metrics['0_clean'] = silhouette_clean
    metrics['0_backdoor'] = silhouette_backdoor
    
    print(f"Class 0 Clean {prefix} silhouette: {silhouette_clean:.4f}")
    print(f"Class 0 Backdoor {prefix} silhouette: {silhouette_backdoor:.4f}")

    return metrics

def analyze_and_visualize_umap3D(data, labels, is_backdoor, epoch, outputs_dir, prefix='u'):
    """
    主函数，执行 UMAP 降维 (3D)、计算聚类指标并绘制可视化图。
    
    参数:
    - data: 原始特征表示 (Tensor)
    - labels: 样本的类别标签 (Tensor)
    - is_backdoor: 是否是 backdoor 数据的标记 (Tensor)
    - epoch: 当前 epoch
    - outputs_dir: 图像保存路径
    - prefix: 图像文件名前缀
    """
    # 转换数据为 numpy 格式
    labels_np = labels.cpu().numpy()
    is_backdoor_np = is_backdoor.cpu().numpy()

    # 使用 UMAP 将数据降维到 3D
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    data_np = data.cpu().numpy()
    # data_umap = reducer.fit_transform(data_np)
    data_umap = batch_umap(data_np, reducer, batch_size=1000)

    plot_umap_3d(data_umap, labels_np, is_backdoor_np, epoch, outputs_dir, prefix)
    
    # 保存降维结果（仅在特定 epoch 保存）
    if epoch in [5, 60, 120]:
        np.save(os.path.join(outputs_dir, f'umap_{prefix}_epoch_{epoch}_3d.npy'), data_umap)
        np.save(os.path.join(outputs_dir, f'labels_{prefix}_epoch_{epoch}.npy'), labels_np)
        np.save(os.path.join(outputs_dir, f'is_backdoor_{prefix}_epoch_{epoch}.npy'), is_backdoor_np)

    # 计算每个类别的指标
    metrics = {}
    silhouette_vals = silhouette_samples(data_umap, labels_np)

    print("### Per-Class Clustering Metrics ###")
    for class_id in range(10):
        # 提取目标类别的轮廓系数
        class_indices = np.where(labels_np == class_id)[0]
        silhouette_mean = np.mean(silhouette_vals[class_indices])
        metrics[class_id] = silhouette_mean
        print(f"Class {class_id} {prefix} silhouette: {silhouette_mean:.4f}")

    # 计算类别 0 中 clean 和 backdoor 数据的指标
    print("### Class 0 Subgroup Metrics ###")
    # 将 backdoor 和 clean 作为独立类别
    extended_labels = labels_np.copy()
    extended_labels[(labels_np == 0) & (is_backdoor_np == 0)] = 10  # clean -> 类别 10
    extended_labels[(labels_np == 0) & (is_backdoor_np == 1)] = 11  # backdoor -> 类别 11
    silhouette_vals = silhouette_samples(data_umap, extended_labels)

    class0_clean_indices = np.where(extended_labels == 10)[0]
    class0_backdoor_indices = np.where(extended_labels == 11)[0]

    silhouette_clean = np.mean(silhouette_vals[class0_clean_indices])
    silhouette_backdoor = np.mean(silhouette_vals[class0_backdoor_indices])

    metrics['0_clean'] = silhouette_clean
    metrics['0_backdoor'] = silhouette_backdoor
    
    print(f"Class 0 Clean {prefix} silhouette: {silhouette_clean:.4f}")
    print(f"Class 0 Backdoor {prefix} silhouette: {silhouette_backdoor:.4f}")

    return metrics
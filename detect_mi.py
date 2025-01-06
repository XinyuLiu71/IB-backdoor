import numpy as np

base_dir = 'results/label_consistent/ob_infoNCE_11_03_0.1_0.6+0.5'

mi_inputs_vs_outputs_dict = {}
mi_Y_vs_outputs_dict = {}

for class_idx in range(10):
    mi_inputs_vs_outputs_dict[class_idx] = np.load(f'{base_dir}/infoNCE_MI_I(X,T)_class_{class_idx}.npy', allow_pickle=True)
    mi_Y_vs_outputs_dict[class_idx] = np.load(f'{base_dir}/infoNCE_MI_I(Y,T)_class_{class_idx}.npy', allow_pickle=True)

    if class_idx == 0:
        mi_inputs_vs_outputs_dict['0_backdoor'] = np.load(f'{base_dir}/infoNCE_MI_I(X,T)_class_0_backdoor.npy', allow_pickle=True)
        mi_Y_vs_outputs_dict['0_backdoor'] = np.load(f'{base_dir}/infoNCE_MI_I(Y,T)_class_0_backdoor.npy', allow_pickle=True)
        mi_inputs_vs_outputs_dict['0_clean'] = np.load(f'{base_dir}/infoNCE_MI_I(X,T)_class_0_clean.npy', allow_pickle=True)
        mi_Y_vs_outputs_dict['0_clean'] = np.load(f'{base_dir}/infoNCE_MI_I(Y,T)_class_0_clean.npy', allow_pickle=True)

for class_idx, mi_inputs_vs_outputs in mi_inputs_vs_outputs_dict.items():
    mi_inputs_vs_outputs_dict[class_idx] = [np.mean(mi_estimate[-5:]) for mi_estimate in mi_inputs_vs_outputs]

for class_idx, mi_Y_vs_outputs in mi_Y_vs_outputs_dict.items():
    mi_Y_vs_outputs_dict[class_idx] = [np.mean(mi_estimate[-5:]) for mi_estimate in mi_Y_vs_outputs]

# 初始化MSE字典
MSE_inputs_vs_outputs = {}
MSE_Y_vs_outputs = {}

# 计算每个类别与其他类别的MSE
for class_idx in range(10):
    n_points = len(mi_inputs_vs_outputs_dict[class_idx])
    total_mse = 0.0
    
    # 与其他类别比较
    for other_class in range(10):
        if other_class == class_idx:
            continue
            
        # 在每个采样点计算MSE
        for i in range(n_points):
            total_mse += (mi_inputs_vs_outputs_dict[class_idx][i] - 
                         mi_inputs_vs_outputs_dict[other_class][i]) ** 2
    
    # 计算平均MSE (除以比较的类别数和采样点数)
    MSE_inputs_vs_outputs[class_idx] = total_mse / (9 * n_points)  # 9是其他类别的数量

# 对Y vs outputs做相同的计算
for class_idx in range(10):
    n_points = len(mi_Y_vs_outputs_dict[class_idx])
    total_mse = 0.0
    
    for other_class in range(10):
        if other_class == class_idx:
            continue
            
        for i in range(n_points):
            total_mse += (mi_Y_vs_outputs_dict[class_idx][i] - 
                         mi_Y_vs_outputs_dict[other_class][i]) ** 2
    
    MSE_Y_vs_outputs[class_idx] = total_mse / (9 * n_points)

print(MSE_inputs_vs_outputs)
print(MSE_Y_vs_outputs)
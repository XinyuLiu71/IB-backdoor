import numpy as np

data1_dir = "results/imagenet10/blend/5.731_0.1_0.4+0.4"
data1_xt = np.load(f"{data1_dir}/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
# data1_ty = np.load(f"{data1_dir}/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()
# data1_xt_r = np.load(f"{data1_dir}/infoNCE_MI_I(X,T)_r.npy", allow_pickle=True).item()

data2_dir = "results/imagenet10/blend/5.732_0.1_0.4+0.4"
data2_xt = np.load(f"{data2_dir}/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
# data2_ty = np.load(f"{data2_dir}/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()

# print(np.array(data1_xt['0_backdoor']).shape)

# data1_xt['0_backdoor'] = data2_xt['0_backdoor']
# data1_xt['0_sample'] = data2_xt['0_sample']

classes = [0, '0_backdoor', '0_clean', '0_sample', 1, 2, 3]
data = {class_idx: [] for class_idx in classes}
for key in data:
    if key in data1_xt:
        data[key].extend(data1_xt[key])
    if key in data2_xt:
        data[key].extend(data2_xt[key])

print(np.array(data['0_backdoor']).shape)
print(data.keys())
np.save(f"{data2_dir}/infoNCE_MI_I(X,T)_r.npy", data)
# np.save(f"{data1_dir}/infoNCE_MI_I(X,T)_r.npy", data1_xt)
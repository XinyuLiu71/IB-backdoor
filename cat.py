import numpy as np

dir1 = "results/cifar10/blend/ob_infoNCE_13_28_0.1_0.1+0.1"
dir2 = "results/cifar10/blend/ob_infoNCE_13_281_0.1_0.1+0.1"

XT1 = np.load(dir1 + "/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
TY1 = np.load(dir1 + "/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()

XT2 = np.load(dir2 + "/infoNCE_MI_I(X,T).npy", allow_pickle=True).item()
TY2 = np.load(dir2 + "/infoNCE_MI_I(Y,T).npy", allow_pickle=True).item()

XT = {}
TY = {}

for class_idx, mi_values in XT1.items():
    XT[class_idx] = np.concatenate([mi_values[:-1], XT2[class_idx]])

for class_idx, mi_values in TY1.items():
    TY[class_idx] = np.concatenate([mi_values[:-1], TY2[class_idx]])

np.save(dir1 + "/infoNCE_MI_I(X,T)_c.npy", XT)
np.save(dir1 + "/infoNCE_MI_I(Y,T)_c.npy", TY)
 
import numpy as np

data = np.load("results/ob_infoNCE_10_22_0.1/suspicious_samples_by_outputs-vs-Y_epoch_1.npy").astype(int)
classes = data[:, 0]
classes_count = {}
for i in classes:
    if i not in classes_count:
        classes_count[i] = 1
    else:
        classes_count[i] += 1
print(classes_count)

indexes = data[np.where(data[:, 0] == 0)][:, 1]
poison_count = 5000
correct = 0
for i in indexes:
    if i<poison_count:
        correct += 1
fp = classes_count[0] - correct
print(f"Detected num: {correct}")
print(f"Detection accuracy for class 0: {correct/5000}%")
print(f"False positive rate for class 0: {fp/5000}%")
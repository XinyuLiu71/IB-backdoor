from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze
import torch
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载管道
image_pipeline = [
    ToTensor(),
    ToDevice(device),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
]

label_pipeline = [
    IntDecoder(),
    ToTensor(),
    ToDevice(device),
    Squeeze()
]

is_backdoor_pipeline = [
    IntDecoder(),
    ToTensor(),
    ToDevice(device),
    Squeeze()
]

def load_and_check_data(batch_size):
    train_dataloader = Loader(
        'data/badnet/0.1/train_data.beton',
        batch_size=batch_size,
        num_workers=1,
        os_cache=True,
        order=OrderOption.RANDOM,
        drop_last=False,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline,
            'is_backdoor': is_backdoor_pipeline
        }
    )
    
    # 创建固定大小的张量来存储数据
    total_samples = 50000
    all_images = torch.zeros((total_samples, 3, 32, 32), device=device)
    all_labels = torch.zeros(total_samples, dtype=torch.long, device=device)
    all_is_backdoor = torch.zeros(total_samples, dtype=torch.long, device=device)
    
    print(f"\nLoading data with batch_size={batch_size}")
    current_idx = 0
    for batch in train_dataloader:
        batch_size = len(batch[1])
        # 按顺序填充数据
        all_images[current_idx:current_idx + batch_size] = batch[0]
        all_labels[current_idx:current_idx + batch_size] = batch[1]
        all_is_backdoor[current_idx:current_idx + batch_size] = batch[2]
        current_idx += batch_size
        
        print(f"Loaded batch with first 5 labels: {batch[1][:5]}")
    
    print(f"Total samples loaded: {current_idx}")
    
    # 检查数据分布
    print(f"\nChecking data with batch_size={batch_size}")
    print(f"Total samples: {len(all_labels)}")
    
    for i in range(10):
        class_mask = (all_labels == i)
        backdoor_mask = (all_labels == i) & (all_is_backdoor == 1)
        print(f"\nClass {i}:")
        print(f"Total samples: {class_mask.sum().item()}")
        print(f"Backdoor samples: {backdoor_mask.sum().item()}")
        print(f"Clean samples: {(class_mask & (all_is_backdoor == 0)).sum().item()}")
    
    # 检查数据的连续性
    unique_labels = torch.unique(all_labels, return_counts=True)
    print("\nUnique labels and their counts:")
    for label, count in zip(unique_labels[0], unique_labels[1]):
        print(f"Label {label}: {count} samples")

# 测试不同的 batch_size
batch_sizes = [1000, 5000, 50000]
for bs in batch_sizes:
    load_and_check_data(bs)
    
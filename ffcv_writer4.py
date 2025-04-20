import os
import argparse
import random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import TensorDataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, TorchTensorField

device = 'cpu'

class DatasetConfig:
    """Container for dataset configuration parameters"""
    def __init__(self, args):
        self.train_path = args.train_data_path
        self.test_path = args.test_data_path
        self.output_dir = Path(args.output_path)
        self.dataset_type = args.dataset
        self.observe_classes = args.observe_classes
        self.poison_rate = args.poison_rate

def load_npz_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate .npz data file
    
    Args:
        path: Path to .npz data file
        
    Returns:
        images: Image data array (N, H, W, C)
        labels: Label array (N,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file {path} not found")
    
    data = np.load(path)
    return data['arr_0'], data['arr_1']

def create_tensor_dataset(images: np.ndarray,
                         labels: np.ndarray,
                         is_backdoor: torch.Tensor = None,
                         device: str = 'cpu') -> TensorDataset:
    """Create standardized TensorDataset with optional backdoor flag
    
    Args:
        images: Image data (N, H, W, C)
        labels: Label data (N,)
        is_backdoor: Optional backdoor flag tensor (N,)
        device: Device for tensor storage
        
    Returns:
        TensorDataset containing images, labels, and optional backdoor flag
    """
    # Create base tensors
    image_tensor = torch.tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    # Add backdoor flag if provided
    if is_backdoor is not None:
        return TensorDataset(image_tensor, label_tensor, is_backdoor)
    
    return TensorDataset(image_tensor, label_tensor)


def create_balanced_sample(backdoor_ds: TensorDataset,
                          clean_ds: TensorDataset,
                          reference_size: int) -> TensorDataset:
    """Create balanced sample dataset
    
    Args:
        backdoor_ds: Poisoned dataset
        clean_ds: Clean dataset
        reference_size: Target dataset size
        
    Returns:
        Balanced combination dataset
    """
    backdoor_size = min(len(backdoor_ds), reference_size // 2)
    clean_size = reference_size - backdoor_size
    
    backdoor_indices = random.sample(range(len(backdoor_ds)), backdoor_size)
    clean_indices = random.sample(range(len(clean_ds)), clean_size)
    
    return TensorDataset(
        torch.cat([
            backdoor_ds.tensors[0][backdoor_indices],
            clean_ds.tensors[0][clean_indices]
        ]),
        torch.cat([
            backdoor_ds.tensors[1][backdoor_indices],
            clean_ds.tensors[1][clean_indices]
        ])
    )

def create_class_datasets(train_images: np.ndarray,
                         train_labels: np.ndarray,
                         config: DatasetConfig) -> Dict[str, TensorDataset]:
    """Create class-specific datasets
    
    Args:
        train_images: Training image data
        train_labels: Training label data
        config: Dataset configuration
        
    Returns:
        Dictionary containing:
        - Standard class datasets
        - Specialized class 0 datasets (poisoned/clean/sampled)
    """
    datasets = {}
    
    # Create standard class datasets
    for cls in config.observe_classes:
        mask = (train_labels == cls)
        datasets[f'class_{cls}'] = create_tensor_dataset(
            train_images[mask], train_labels[mask])
    
    # Special handling for class 0
    if 0 in config.observe_classes:
        cls0_mask = (train_labels == 0)
        cls0_images = train_images[cls0_mask]
        cls0_labels = train_labels[cls0_mask]
        
        # Split poisoned and clean data
        poison_num = int(len(train_images) * config.poison_rate)
        datasets.update({
            'class_0_backdoor': create_tensor_dataset(
                cls0_images[:poison_num], cls0_labels[:poison_num]),
            'class_0_clean': create_tensor_dataset(
                cls0_images[poison_num:], cls0_labels[poison_num:])
        })
        
        # Create balanced sample dataset
        datasets['class_0_sample'] = create_balanced_sample(
            datasets['class_0_backdoor'],
            datasets['class_0_clean'],
            reference_size=len(datasets.get('class_1', datasets['class_0']))
        )
    
    return datasets


def write_beton_dataset(dataset: TensorDataset,
                       output_path: Path,
                       fields: Dict,
                       dataset_name: str = ''):
    """Write dataset to Beton format
    
    Args:
        dataset: Dataset to write
        output_path: Output directory
        fields: Field definitions
        dataset_name: Dataset name for logging
    """
    writer = DatasetWriter(output_path, fields)
    writer.from_indexed_dataset(dataset)
    print(f"Dataset successfully written: {dataset_name} -> {output_path}, num_samples={len(dataset)}")

def main(config: DatasetConfig):
    """Main processing pipeline"""
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    train_images, train_labels = load_npz_data(config.train_path)
    test_images, test_labels = load_npz_data(config.test_path)
    
    # Create backdoor flags for training data
    total_samples = len(train_images)
    poison_samples = int(total_samples * config.poison_rate)
    is_backdoor = torch.zeros(total_samples, dtype=torch.long, device=device)
    is_backdoor[:poison_samples] = 1
    
    # Create base datasets
    base_datasets = {
        'train': create_tensor_dataset(train_images, train_labels, is_backdoor),
        'test': create_tensor_dataset(test_images, test_labels)
    }
    
    # Create class-specific datasets
    class_datasets = create_class_datasets(train_images, train_labels, config)
    
    # Define common field templates
    COMMON_FIELDS = {
        'image': TorchTensorField(dtype=torch.float32, shape=(3, 224, 224)), # image size depends on the dataset
        'label': IntField()
    }
    TRAIN_FIELDS = {
        **COMMON_FIELDS,
        'is_backdoor': IntField()
    }
    
    # Write datasets based on configuration
    if config.dataset_type in ('all', 'train'):
        write_beton_dataset(
            base_datasets['train'],
            config.output_dir/'train_data.beton',
            TRAIN_FIELDS,
            'Training dataset'
        )
    
    if config.dataset_type in ('all', 'test'):
        write_beton_dataset(
            base_datasets['test'],
            config.output_dir/'test_data.beton',
            COMMON_FIELDS,
            'Test dataset'
        )
    
    if config.dataset_type in ('all', 'sample'):
        for name, dataset in class_datasets.items():
            write_beton_dataset(
                dataset,
                config.output_dir/f'{name}.beton',
                COMMON_FIELDS,
                f'Sample dataset {name}'
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset conversion tool')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to training data (.npz format)')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to test data (.npz format)')
    parser.add_argument('--output_path', type=str, default='datasets',
                        help='Output directory path')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'train', 'test', 'sample'],
                        help='Dataset type to generate')
    parser.add_argument('--observe_classes', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9],
                        help='List of classes to observe')
    parser.add_argument('--poison_rate', type=float, default=0.05,
                        help='Poison data ratio (0-1)')
    
    args = parser.parse_args()
    config = DatasetConfig(args)
    
    try:
        main(config)
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        exit(1)
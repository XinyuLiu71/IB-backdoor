import numpy as np
import os
import argparse
import torch
import torch.nn.functional as F
from copy import deepcopy
from ResNet import ResNet18
import torch.nn as nn
from tqdm import tqdm
from pgd_attack import PgdAttack
from torchvision import transforms
from torchvision import datasets
import random
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import cv2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CONFIGS = {
    'cifar10': {
        'n_classes': 10,
        'img_size': 32,
        'channels': 3,
    },
    'svhn': {
        'n_classes': 10,
        'img_size': 32,
        'channels': 3,
        'sample_size_per_class': 4000
    },
    'imagenet10': {
        'n_classes': 10,
        'img_size': 224,
        'channels': 3,
        'classes': ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
                    'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
    },
}

def get_dataset(name, train=True, download=True, root='../data'):
    if name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.SVHN(
            root=root, split='train' if train else 'test',
            download=download, transform=transform
        )

        # Sample the training set, keeping only the first 4000 samples per class
        if train:
            # Get all labels
            all_labels = dataset.labels
            # Create new data list
            new_data = []
            new_labels = []

            # Process each class
            for label in range(10):
                # Get all indices for current class
                indices = np.where(all_labels == label)[0]
                # Keep only the first 4000 samples
                selected_indices = indices[:DATASET_CONFIGS[name]['sample_size_per_class']]
                # Add selected samples
                new_data.extend([dataset.data[i] for i in selected_indices])
                new_labels.extend([dataset.labels[i] for i in selected_indices])

            # Update dataset
            dataset.data = np.array(new_data)
            dataset.labels = np.array(new_labels)

    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
    elif name == 'imagenet10':
        img_size = DATASET_CONFIGS[name]['img_size']
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

        # Create custom dataset with train/test split
        dataset = ImageNet10LocalDataset(
            root='../imagenet-10',
            train=train,
            transform=transform,
            test_size=0.2,  # You can adjust the test size
            random_state=42  # For reproducibility
        )
    else:
        pass # Add more datasets here
    return dataset

class ImageNet10LocalDataset(torch.utils.data.Dataset):
    """Custom dataset for ImageNet10 from local directory with train/test split"""

    def __init__(self, root, train=True, transform=None, test_size=0.2, random_state=42):
        self.root = root
        self.train = train
        self.transform = transform
        self.classes = [
            'n02056570', 'n02128757', 'n02692877', 'n04254680', 'n04467665',
            'n02085936', 'n02690373', 'n03095699', 'n04285008', 'n07747607'
        ]
        self.samples = []
        self.labels = []

        # Load all images and labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append(img_path)
                    self.labels.append(class_idx)

        print(f"num of samples: {len(self.labels)}")

        # Split the data into train and test sets - each class: first 1000 for train, next 300 for test
        self.train_samples = []
        self.test_samples = []
        self.train_labels = []
        self.test_labels = []
        
        for class_idx in range(len(self.classes)):
            # Get all samples for this class
            class_indices = [i for i, label in enumerate(self.labels) if label == class_idx]
            class_samples = [self.samples[i] for i in class_indices]
            
            total_samples = len(class_samples)
            print(f"Class {class_idx}: {total_samples} samples")
            
            if total_samples < 1300:
                print(f"Warning: Class {class_idx} only has {total_samples} samples, less than required 1300")
                # If not enough samples, use all available
                train_samples = class_samples[:min(1000, total_samples)]
                test_samples = class_samples[1000:min(1300, total_samples)] if total_samples > 1000 else []
            else:
                # Split: first 1000 for train, next 300 for test
                train_samples = class_samples[:1000]
                test_samples = class_samples[1000:1300]  # Next 300 samples
            
            self.train_samples.extend(train_samples)
            self.test_samples.extend(test_samples)
            self.train_labels.extend([class_idx] * len(train_samples))
            self.test_labels.extend([class_idx] * len(test_samples))
        
        print(f"Train samples: {len(self.train_samples)}, Test samples: {len(self.test_samples)}")
        
        # Display class distribution
        if train:
            train_class_counts = {}
            for label in self.train_labels:
                train_class_counts[label] = train_class_counts.get(label, 0) + 1
            print("Train label distribution:")
            for class_idx in sorted(train_class_counts.keys()):
                print(f"Label {class_idx}: {train_class_counts[class_idx]}")
        else:
            test_class_counts = {}
            for label in self.test_labels:
                test_class_counts[label] = test_class_counts.get(label, 0) + 1
            print("Test label distribution:")
            for class_idx in sorted(test_class_counts.keys()):
                print(f"Label {class_idx}: {test_class_counts[class_idx]}")

    def __len__(self):
        return len(self.train_samples) if self.train else len(self.test_samples)

    def __getitem__(self, idx):
        if self.train:
            img_path = self.train_samples[idx]
            label = self.train_labels[idx]
        else:
            img_path = self.test_samples[idx]
            label = self.test_labels[idx]

        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label

# Get trigger mask
def get_trigger_mask(img_size, total_pieces, masked_pieces):
    div_num = int(np.sqrt(total_pieces))
    step = int(img_size // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = np.ones((img_size, img_size, 1))
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[y * step: (y + 1) * step, x * step: (x + 1) * step] = 0
    return mask

class TriggerGenerator:
    """Generates different types of backdoor triggers for image poisoning"""
    
    @staticmethod
    def blend_trigger(image: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Apply blend trigger using alpha blending
        Args:
            image: Original image in [H, W, C] format
            mask: Trigger pattern to blend
            alpha: Blending ratio (0-1)
        Returns:
            Poisoned image with blended trigger
        """
        # if mask.shape[-1] != image.shape[-1]:
        #     mask = np.repeat(mask[:, :, np.newaxis], image.shape[-1], axis=2)
        poisoned_image = (1 - alpha) * image + alpha * mask
        return poisoned_image
    
    @staticmethod
    def adaptive_blend_trigger(image: np.ndarray, mask: np.ndarray, trigger_mask: np.ndarray, alpha: float = 0.15) -> np.ndarray:
        """
        Apply adaptive blend trigger using alpha blending with partial masking
        Args:
            image: Original image in [H, W, C] format
            mask: Trigger pattern to blend
            trigger_mask: Mask indicating which parts of the trigger to apply
            alpha: Blending ratio (0-1), smaller than standard blend for training
        Returns:
            Poisoned image with adaptively blended trigger
        """
        # Apply trigger mask to determine which areas to apply the trigger
        effective_mask = mask * trigger_mask
        poisoned_image = (1 - alpha) * image + alpha * effective_mask
        return poisoned_image

    @staticmethod
    def badnet_trigger(image: np.ndarray, trigger_size=5) -> np.ndarray:
        """
        Apply BadNets-style trigger (3x3 red square in corner)
        Args:
            image: Original image in [H, W, C] format
        Returns:
            Poisoned image with red square trigger
        """
        poisoned_image = image.copy()
        poisoned_image[:trigger_size, :trigger_size, 0] = 1  # Red channel
        poisoned_image[:trigger_size, :trigger_size, 1] = 0  # Green channel
        poisoned_image[:trigger_size, :trigger_size, 2] = 0  # Blue channel
        return poisoned_image

    @staticmethod
    def wanet_trigger(
        image: np.ndarray, 
        identity_grid: torch.Tensor, 
        noise_grid: torch.Tensor, 
        s: float = 0.5, 
        noise: bool = False
    ) -> np.ndarray:
        """
        Apply WaNet-style spatial transformation trigger
        Args:
            image: Original image in [H, W, C] format
            identity_grid: Base grid for transformation
            noise_grid: Noise pattern grid
            s: Scaling factor for noise
            noise: Whether to add random noise
        Returns:
            Poisoned image with spatial transformation
        """
        h = identity_grid.shape[2]
        grid = identity_grid + s * noise_grid / h
        grid = torch.clamp(grid, -1, 1)
        
        if noise:
            ins = torch.rand(1, h, h, 2) * 2 - 1
            grid = torch.clamp(grid + ins / h, -1, 1)
        
        # Convert image to tensor and apply grid sample
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1, 2)
        poisoned_image = F.grid_sample(
            image_tensor, grid, align_corners=True
        ).squeeze().permute(1, 2, 0).numpy()
        
        return poisoned_image

    @staticmethod
    def label_consistent_trigger(image: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """
        Apply label-consistent trigger with corner perturbations
        Args:
            image: Original image in [H, W, C] format
            amplitude: Perturbation strength
        Returns:
            Poisoned image with corner triggers
        """
        poisoned_image = image.astype(np.float32)
        trigger_pattern = np.array([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]], dtype=np.uint8)
        
        # Expand pattern for RGB if needed
        if image.shape[2] == 3:
            trigger_pattern = np.stack([trigger_pattern]*3, axis=-1)
        
        perturbation = np.where(trigger_pattern == 1, amplitude, -amplitude)
        h, w, _ = poisoned_image.shape
        
        # Apply to four corners
        corners = [
            (slice(0,3), slice(0,3)),
            (slice(0,3), slice(w-3,w)),
            (slice(h-3,h), slice(0,3)),
            (slice(h-3,h), slice(w-3,w))
        ]
        
        for y_slice, x_slice in corners:
            poisoned_image[y_slice, x_slice] = np.clip(
                poisoned_image[y_slice, x_slice] + perturbation, 0, 1
            )
            
        return poisoned_image
    
    @staticmethod
    def ftrojan_trigger(image: np.ndarray, freq_coords=[(15, 15), (31, 31)], magnitude=30) -> np.ndarray:
        """
        Apply FTrojan-style frequency domain trigger (invisible DCT UV perturbation)
        Args:
            image: RGB image in [H, W, C] format, float32 in [0, 1]
        Returns:
            Poisoned image with frequency domain trigger
        """
        image = (image * 255).astype(np.uint8)

        # Convert to YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        
        for ch in [1, 2]:  # Cr and Cb (UV channels)
            dct_map = cv2.dct(yuv[:, :, ch])
            for (k1, k2) in freq_coords:
                dct_map[k1, k2] += magnitude
            yuv[:, :, ch] = cv2.idct(dct_map)

        poisoned = cv2.cvtColor(np.clip(yuv, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        return poisoned.astype(np.float32) / 255.0


class PoisonDatasetGenerator:
    """Main class for generating poisoned datasets"""
    
    def __init__(
        self, 
        attack_type: str,
        dataset: str,
        target_class: int,
        poison_percentage: float,
        data_dir: str = "../data"
    ):
        """
        Initialize poison generator
        Args:
            attack_type: Type of backdoor attack
            target_class: Target class for poisoning
            poison_percentage: Percentage of data to poison
            data_dir: Root directory for dataset storage
        """
        self.attack_type = attack_type
        self.dataset = dataset
        self.target_class = target_class
        self.poison_percentage = poison_percentage
        self.output_dir = os.path.join(data_dir, dataset, attack_type, str(poison_percentage))
        self.trigger_generator = TriggerGenerator()
        
        # Attack-specific initialization
        if attack_type == 'wanet':
            self._init_wanet_params()
        elif attack_type == 'label_consistent':
            self._init_label_consistent_attack()
        elif attack_type == 'adaptive_blend':
            self.conservatism_ratio = 0.5  # Default: 50% of poisoned samples keep original labels
            self.pieces = 16  # Default: divide trigger into 4x4=16 blocks
            self.mask_rate = 0.5  # Default: randomly mask 50% of trigger blocks
            self.masked_pieces = int(self.mask_rate * self.pieces)
            self.train_alpha = 0.15  # Use lower opacity during training
            self.test_alpha = 0.2   # Use higher opacity during testing

    def _init_wanet_params(self):
        """Initialize WaNet-specific grid parameters"""
        k = 4  # Noise grid size
        input_size = DATASET_CONFIGS[self.dataset]['img_size']
        ins = torch.rand(1, 2, k, k) * 2 - 1 # [-1, 1]
        ins = ins / torch.mean(torch.abs(ins)) # normalize
        # Initialize noise grid
        self.noise_grid = F.interpolate(
            ins,
            size=input_size,
            mode="bicubic",
            align_corners=True
        ).permute(0, 2, 3, 1)
        
        # Create identity grid
        linspace = torch.linspace(-1, 1, steps=input_size)
        x, y = torch.meshgrid(linspace, linspace)
        self.identity_grid = torch.stack((y, x), 2)[None, ...]

    def _init_label_consistent_attack(self):
        """Initialize label-consistent attack components"""
        # Attack parameters
        self.eps = 300 / 255.0
        self.alpha = 1.5 / 255.0
        self.steps = 100
        self.ord = 2  # L2 norm
        
        # Create output directory
        self.adv_dataset_dir = os.path.join(self.output_dir, 'adv_dataset')
        os.makedirs(self.adv_dataset_dir, exist_ok=True)

        # Initialize adversarial model
        self._load_adv_model()

    def _load_adv_model(self):
        """Load pre-trained model for generating adversarial examples"""
        print("Initializing adversarial model...")
        self.adv_model = ResNet18(num_classes=10)
        
        # Adjust model architecture for CIFAR-10
        self.adv_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.adv_model.fc = nn.Linear(512, 10)
        
        try:
            state_dict = torch.load(
                'data/clean/resnet18_cifar10.pt',
                map_location=device,
                weights_only=True
            )
            self.adv_model.load_state_dict(state_dict)
            print("Model weights loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
            
        self.adv_model.to(device).eval()
        self.attacker = PgdAttack(
            self.adv_model, 
            eps=self.eps, 
            steps=self.steps,
            eps_lr=self.alpha, 
            ord=self.ord
        )

    def _load_data(self, data_dir: str = "../data"):
        """Load and preprocess dataset"""
        train_data = get_dataset(self.dataset, train=True, root=data_dir)
        # Convert to numpy arrays and handle the permutation correctly
        train_images = torch.stack([x[0] for x in train_data]).numpy().astype('float32')
        train_labels = torch.tensor([x[1] for x in train_data]).numpy()
        
        test_data = get_dataset(self.dataset, train=False, root=data_dir)
        test_images = torch.stack([x[0] for x in test_data]).numpy().astype('float32')
        test_labels = torch.tensor([x[1] for x in test_data]).numpy()
        
        # Ensure labels are 1D
        train_labels = np.squeeze(train_labels)
        test_labels = np.squeeze(test_labels)
        
        # Convert from [N, C, H, W] to [N, H, W, C] format
        train_images = np.transpose(train_images, (0, 2, 3, 1))
        test_images = np.transpose(test_images, (0, 2, 3, 1))
        
        return (train_images, train_labels), (test_images, test_labels)

    def add_trigger(self, image, amplitude=0, is_test=False):
        if self.attack_type == 'label_consistent':
            # Process input image
            image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            # Process target class - modify this part
            target = torch.tensor([self.target_class], dtype=torch.long).to(device)  # Add batch dimension and move to same device
            # Perform adversarial attack
            adv_image = self.attacker.perturb(image_tensor, target)
            # Convert back to numpy format
            adv_image = adv_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            return self.trigger_generator.label_consistent_trigger(adv_image, amplitude)
        elif self.attack_type == 'blend':
            if self.dataset == 'imagenet10':
                mask = np.load('trigger/Blendnoise_224.npy') / 255.0
                # trigger = Image.open('trigger/hellokitty_224.png').convert('RGB')
                # mask = np.array(trigger) / 255.0
            else:
                mask = np.load('trigger/Blendnoise.npy') / 255.0
            mask = np.expand_dims(mask, axis=-1)
            return self.trigger_generator.blend_trigger(image, mask)
        elif self.attack_type == 'adaptive_blend':
            mask = np.load('trigger/Blendnoise.npy') / 255.0
            mask = np.expand_dims(mask, axis=-1)
            if is_test:
                # Use complete trigger and higher opacity for testing
                return self.trigger_generator.blend_trigger(image, mask, alpha=self.test_alpha)
            else:
                # Use partial trigger and lower opacity for training
                img_size = DATASET_CONFIGS[self.dataset]['img_size']
                trigger_mask = get_trigger_mask(img_size, self.pieces, self.masked_pieces)
                return self.trigger_generator.adaptive_blend_trigger(image, mask, trigger_mask, alpha=self.train_alpha)
        elif self.attack_type == 'badnet':
            return self.trigger_generator.badnet_trigger(image, trigger_size=30 if self.dataset == 'imagenet10' else 5)
        elif self.attack_type == 'wanet':
            return self.trigger_generator.wanet_trigger(image, self.identity_grid, self.noise_grid, noise=True)
        elif self.attack_type == 'ftrojan':
            return self.trigger_generator.ftrojan_trigger(image, freq_coords=[(15, 15), (31, 31)], magnitude=30)


    def generate_poisoned_dataset(self):
        (train_images, train_labels), (test_images, test_labels) = self._load_data()
        n_classes = DATASET_CONFIGS[self.dataset]['n_classes']
        samples_per_class = len(train_images) // n_classes

        if self.attack_type == 'label_consistent':
            poison_count = int(self.poison_percentage * samples_per_class) # n% of target class
            
            # Create a copy of the training data
            blend_images = train_images.copy()
            blend_labels = train_labels.copy()
            
            # Get indices of target class samples
            target_indices = np.where(blend_labels == self.target_class)[0][:poison_count]
            
            # Poison the selected samples and show comparisons for first few images
            for idx in tqdm(target_indices):
                # original_image = blend_images[idx].copy()
                blend_images[idx] = self.add_trigger(blend_images[idx], amplitude=64/255)
                
                # Show comparison for first 5 images
                # self.display_comparison(original_image, blend_images[idx], idx)
            poisoned_test_images = test_images.copy()
            poisoned_test_images = np.array([self.trigger_generator.label_consistent_trigger(img, amplitude=1) for img in poisoned_test_images])

        elif self.attack_type == 'adaptive_blend':
            poison_count = int(self.poison_percentage * len(train_images))  # n% of train dataset
            clean_data_num = samples_per_class - int(poison_count // n_classes)
            class_0_clean = train_images[train_labels == 0][:clean_data_num]
            
            poison_classes = np.arange(0, n_classes)
            poison_images = []
            poison_labels = []  # New: record labels for poisoned samples

            for _class in poison_classes:
                # Get samples from this class for poisoning
                imgs = train_images[train_labels == _class][-int(poison_count / n_classes):]
                imgs_labels = np.full(len(imgs), _class, dtype=np.int64)
                
                # Calculate number of samples to keep original labels and payload samples
                samples_per_poison_class = len(imgs)
                regularization_count = int(samples_per_poison_class * self.conservatism_ratio)
                payload_count = samples_per_poison_class - regularization_count
                print(f"Class {_class}: {samples_per_poison_class} samples, payload_count: {payload_count}, regularization_count: {regularization_count}")
                # For each sample
                for idx in range(imgs.shape[0]):
                    # Add adaptive trigger
                    imgs[idx] = self.add_trigger(imgs[idx])
                    
                    # First payload_count samples change label to target class, others keep original label
                    if idx < payload_count:
                        imgs_labels[idx] = self.target_class
                    # Remaining regularization_count samples keep original label
                
                poison_images.append(imgs)
                poison_labels.append(imgs_labels)

            # Combine all poisoned samples
            merged_poison_images = np.concatenate(poison_images, axis=0)
            merged_poison_labels = np.concatenate(poison_labels, axis=0)

            # Prepare clean data for classes 1-9
            clean_images, clean_labels = self.prepare_clean_data(train_images, train_labels, clean_data_num)

            # Combine poisoned and clean data
            blend_images = np.concatenate([merged_poison_images, class_0_clean, clean_images], axis=0)
            blend_labels = np.concatenate([merged_poison_labels, np.zeros(len(class_0_clean)), clean_labels], axis=0)

            # Process test set images
            poisoned_test_images = test_images.copy()
            poisoned_test_images = np.array([self.add_trigger(img, is_test=True) for img in tqdm(poisoned_test_images)])
        else:
            poison_count = int(self.poison_percentage * len(train_images)) # n% of train dataset
            
            clean_data_num = samples_per_class - int(poison_count // n_classes)
            # print(f"sample_per_classes: {samples_per_class}, clean_data_num: {clean_data_num}, poison_count: {poison_count}")
            class_0_clean = train_images[train_labels == 0][:clean_data_num]
            poison_classes = np.arange(0, n_classes)
            poison_images = []

            for _class in poison_classes:
                img = train_images[train_labels == _class][-int(poison_count / n_classes):]
                for idx in range(img.shape[0]):
                    img[idx] = self.add_trigger(img[idx])
                poison_images.append(img)

            merged_poison_images = np.concatenate(poison_images, axis=0)
            # Prepare clean data for labels 1-9
            clean_images, clean_labels = self.prepare_clean_data(train_images, train_labels, clean_data_num)

            # Combine poisoned and clean data
            blend_images = np.concatenate([np.concatenate(poison_images), class_0_clean, clean_images], axis=0)
            blend_labels = np.hstack([np.zeros(len(np.concatenate(poison_images)) + len(class_0_clean)), clean_labels])

            # Add trigger to test images and save
            poisoned_test_images = test_images.copy()
            poisoned_test_images = np.array([self.add_trigger(img, amplitude=1) for img in tqdm(poisoned_test_images)])
    
        # Save the final dataset
        self.save_data(blend_images, blend_labels, f'{self.attack_type}_{self.poison_percentage}.npz')
        self.save_data(test_images, test_labels, 'test_data.npz')
        self.save_data(poisoned_test_images, test_labels, 'poisoned_test_data.npz')
        # Display sample poisoned images
        # self.display_poison_images(poison_images, poison_classes)

    def prepare_clean_data(self, train_images, train_labels, clean_data_num):
        n_classes = DATASET_CONFIGS[self.dataset]['n_classes']
        clean_classes = np.arange(1, n_classes)
        clean_images = []
        clean_labels = []
        for _class in clean_classes:
            img = train_images[train_labels == _class][:clean_data_num]
            clean_images.append(img)
            clean_labels.append([_class] * img.shape[0])
        return np.concatenate(clean_images, axis=0), np.concatenate(clean_labels, axis=0)

    def save_data(self, images, labels, filename):
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, images, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', type=str, choices=['blend', 'badnet', 'wanet', 'label_consistent', 'adaptive_blend', 'ftrojan'], required=False, default='badnet', help='Type of attack')
    parser.add_argument('--dataset', type=str, default='imagenet10', choices=['cifar10', 'svhn', 'imagenet10'],help='dataset name')
    parser.add_argument('--target_class', type=int, default=0, help='Target class for attack')
    parser.add_argument('--poison_percentage', type=float, default=0.1, help='Percentage of poisoned data')
    parser.add_argument('--data_dir', type=str, default="../data", help='Data directory')
    parser.add_argument('--conservatism_ratio', type=float, default=0.5, help='Ratio of poisoned samples that keep original labels (for adaptive_blend)')
    args = parser.parse_args()

    generator = PoisonDatasetGenerator(
        args.attack_type,
        args.dataset,
        args.target_class,
        args.poison_percentage,
        args.data_dir
    )
    
    # If using adaptive blend attack, set conservatism_ratio
    if args.attack_type == 'adaptive_blend':
        generator.conservatism_ratio = args.conservatism_ratio
        
    generator.generate_poisoned_dataset()
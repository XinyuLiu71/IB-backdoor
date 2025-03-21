import numpy as np
import os
import argparse
from tensorflow.keras.datasets import cifar10
import torch
import torch.nn.functional as F
from copy import deepcopy
from ResNet import ResNet18
import torch.nn as nn
from tqdm import tqdm
from pgd_attack import PgdAttack

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        poisoned_image = (1 - alpha) * image + alpha * mask
        return poisoned_image

    @staticmethod
    def badnet_trigger(image: np.ndarray) -> np.ndarray:
        """
        Apply BadNets-style trigger (3x3 red square in corner)
        Args:
            image: Original image in [H, W, C] format
        Returns:
            Poisoned image with red square trigger
        """
        poisoned_image = image.copy()
        poisoned_image[:5, :5, 0] = 1  # Red channel
        poisoned_image[:5, :5, 1] = 0  # Green channel
        poisoned_image[:5, :5, 2] = 0  # Blue channel
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


class PoisonDatasetGenerator:
    """Main class for generating poisoned datasets"""
    
    def __init__(
        self, 
        attack_type: str,
        dataset: str,
        target_class: int,
        poison_percentage: float,
        data_dir: str = "data"
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
        self.target_class = target_class
        self.poison_percentage = poison_percentage
        self.output_dir = os.path.join(data_dir, dataset, attack_type, str(poison_percentage))
        self.trigger_generator = TriggerGenerator()
        
        # Attack-specific initialization
        if attack_type == 'wanet':
            self._init_wanet_params()
        elif attack_type == 'label_consistent':
            self._init_label_consistent_attack()

    def _init_wanet_params(self):
        """Initialize WaNet-specific grid parameters"""
        k = 4  # Noise grid size
        input_size = 32  # CIFAR-10 image size
        
        # Initialize noise grid
        self.noise_grid = F.interpolate(
            (torch.rand(1, 2, k, k)*2-1) / torch.mean(torch.abs(ins)),
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

    def _load_data(self):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        train_labels = np.squeeze(train_labels)
        test_labels = np.squeeze(test_labels)
        return (train_images, train_labels), (test_images, test_labels)

    def add_trigger(self, image, amplitude=0):
        if self.attack_type == 'label_consistent':
            # 处理输入图像
            image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            # 处理目标类别 - 修改这部分
            target = torch.tensor([self.target_class], dtype=torch.long).to(device)  # 添加batch维度并移至相同设备
            # 进行对抗攻击
            adv_image = self.attacker.perturb(image_tensor, target)
            # 转回numpy格式
            adv_image = adv_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            return self.trigger_generator.label_consistent_trigger(adv_image, amplitude)
            # return adv_image
        elif self.attack_type == 'blend':
            mask = np.load('poison_data_generator/trigger/Blendnoise.npy') / 255.0
            mask = np.expand_dims(mask, axis=-1)
            return self.trigger_generator.blend_trigger(image, mask)
        elif self.attack_type == 'badnet':
            return self.trigger_generator.badnet_trigger(image)
        elif self.attack_type == 'wanet':
            self.setup_wanet()
            return self.trigger_generator.wanet_trigger(image, self.identity_grid, self.noise_grid, noise=True)


    def generate_poisoned_dataset(self):
        (train_images, train_labels), (test_images, test_labels) = self._load_data()

        if self.attack_type == 'label_consistent':
            poison_count = int(self.poison_percentage * 5000)
            
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

        else:
            poison_count = int(self.poison_percentage * 50000)
            clean_data_num = 5000 - int(poison_count / 10)

            class_0_clean = train_images[train_labels == 0][:clean_data_num]
            poison_classes = np.arange(0, 10)
            poison_images = []

            for _class in poison_classes:
                img = train_images[train_labels == _class][-int(poison_count / 10):]
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
        clean_classes = np.arange(1, 10)
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

    # def display_poison_images(self, poison_images, poison_classes):
    #     fig, axes = plt.subplots(len(poison_classes), 3, figsize=(8, 2*len(poison_classes)))
    #     plt.subplots_adjust(wspace=0.05, hspace=0.2)
        
    #     for i, _class in enumerate(poison_classes):
    #         sampled_images = poison_images[i][np.random.choice(poison_images[i].shape[0], 3, replace=False)]
    #         for j in range(3):
    #             img = sampled_images[j]
    #             img = np.clip(img, 0, 1)
    #             axes[i, j].imshow(img)
    #             axes[i, j].axis('off')
            
    #         axes[i, 0].text(-0.1, 1.1, f'Class {_class}', transform=axes[i, 0].transAxes, 
    #                         va='top', ha='right', fontsize=8, fontweight='bold')

    #     plt.tight_layout()
    #     plt.savefig(f"poison_images_sample_{self.attack_type}.png", dpi=300, bbox_inches='tight')
    #     plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', type=str, choices=['blend', 'badnet', 'wanet', 'label_consistent'], required=False, default='badnet', help='Type of attack')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'svhn'], required=False, default='cifar10', help='Dataset')
    parser.add_argument('--target_class', type=int, default=0, help='Target class for attack')
    parser.add_argument('--poison_percentage', type=float, default=0.1, help='Percentage of poisoned data')
    parser.add_argument('--data_dir', type=str, default="../data", help='Data directory')
    args = parser.parse_args()

    generator = PoisonDatasetGenerator(
        args.attack_type,
        args.dataset,
        args.target_class,
        args.poison_percentage,
        args.data_dir
    )
    generator.generate_poisoned_dataset()

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
import matplotlib.pyplot as plt
from pgd_attack import PgdAttack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TriggerGenerator:
    @staticmethod
    def blend_trigger(image, mask, alpha=0.3):
        return (1 - alpha) * image + alpha * mask

    @staticmethod
    def badnet_trigger(image):
        image[:5, :5, 0] = 1  # Red channel
        image[:5, :5, 1] = 0  # Green channel
        image[:5, :5, 2] = 0  # Blue channel
        return image

    @staticmethod
    def wanet_trigger(image, identity_grid, noise_grid, s=0.5, noise=False):
        """
        s: scale factor
        noise: whether add noise
        """
        h = identity_grid.shape[2]
        grid = identity_grid + s * noise_grid / h
        grid = torch.clamp(grid, -1, 1)
        
        if noise:
            ins = torch.rand(1, h, h, 2) * 2 - 1
            grid = torch.clamp(grid + ins / h, -1, 1)
        
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1, 2)
        poisoned_image = F.grid_sample(image_tensor, grid, align_corners=True).squeeze().permute(1, 2, 0).numpy()
        return poisoned_image

    @staticmethod
    def label_consistent_trigger(image, amplitude=1):
        """
        Apply label-consistent trigger to the image with reduced visibility.
        This adds a subtle perturbation based on a 3x3 black-and-white trigger pattern
        in the four corners of the image.
        
        Args:
            image (numpy.ndarray): Original image (H, W, C) in range [0,1].
            amplitude (float): The amplitude of the trigger (default: 30/255).
            
        Returns:
            numpy.ndarray: Poisoned image in range [0,1].
        """
        # Blend the original and adversarial images
        poisoned_image = image.astype(np.float32)

        # Define the 3x3 black and white trigger pattern
        trigger_pattern = np.array([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]], dtype=np.uint8)

        # Convert trigger pattern to 3 channels if needed
        if image.shape[2] == 3:  # Check if it's RGB
            trigger_pattern = np.stack([trigger_pattern] * 3, axis=-1)
        
        # Calculate trigger perturbation based on amplitude
        trigger_perturbation = np.where(trigger_pattern == 1, amplitude, -amplitude)
        # trigger_perturbation = 0

        # Add the reduced visibility trigger to the four corners
        h, w, _ = poisoned_image.shape
        
        # Apply trigger perturbation to each corner
        poisoned_image[:3, :3] = np.clip(poisoned_image[:3, :3] + trigger_perturbation, 0, 1)
        poisoned_image[:3, w-3:w] = np.clip(poisoned_image[:3, w-3:w] + trigger_perturbation, 0, 1)
        poisoned_image[h-3:h, :3] = np.clip(poisoned_image[h-3:h, :3] + trigger_perturbation, 0, 1)
        poisoned_image[h-3:h, w-3:w] = np.clip(poisoned_image[h-3:h, w-3:w] + trigger_perturbation, 0, 1)

        return poisoned_image


class PoisonDataGenerator:
    def __init__(self, attack_type, target_class, poison_percentage, data_dir):
        self.attack_type = attack_type
        self.target_class = target_class
        self.poison_percentage = poison_percentage
        self.output_dir = os.path.join(data_dir, attack_type, str(poison_percentage))
        self.trigger_generator = TriggerGenerator()
        if attack_type == 'wanet':
            self.setup_wanet()
        elif attack_type == 'label_consistent':
            self.setup_label_consistent()

    def setup_wanet(self):
        k = 4 # size of noise grid
        input_height = 32
        input_width = 32
        ins = torch.rand(1, 2, k, k) * 2 - 1 # [-1, 1]
        ins = ins / torch.mean(torch.abs(ins)) # normalize
        # upsample to input size
        self.noise_grid = F.interpolate(ins, size=input_height, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
        array1d = torch.linspace(-1, 1, steps=input_height)
        # generate grid
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...]

    def setup_label_consistent(self):
        self.eps = 300 / 255.0
        self.alpha = 1.5 / 255.0
        self.steps = 100
        self.ord = 2
        self.adv_dataset_dir = os.path.join(self.output_dir, 'adv_dataset')
        os.makedirs(self.adv_dataset_dir, exist_ok=True)

        print("Initializing ResNet18 model...")
        self.adv_model = ResNet18(num_classes=10)
        self.adv_model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.adv_model.fc = torch.nn.Linear(512, 10)

        print("Loading model weights...")
        try:
            # 使用 weights_only=True 来避免pickle相关的警告和潜在问题
            state_dict = torch.load(
                'data/clean/resnet18_cifar10.pt',
                map_location='cpu',
                weights_only=True
            )
            self.adv_model.load_state_dict(state_dict)
            print("Model weights loaded successfully!")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            raise
        self.adv_model.to(device)
        self.adv_model.eval()  # Set model to evaluation mode
        print("Model setup completed!")

        self.attacker = PgdAttack(self.adv_model, eps=self.eps, steps=self.steps, eps_lr=self.alpha, ord=self.ord)

    def load_data(self):
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

    def display_comparison(self, original_image, poisoned_image, index):
        """Display original and poisoned images side by side"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(poisoned_image, 0, 1))
        plt.title('Poisoned Image')
        plt.axis('off')
        
        save_dir = os.path.join(self.output_dir, 'comparison_images')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'comparison_{index}.png'))
        print(f'Comparison image saved as {save_dir}')
        plt.close()

    def generate_poisoned_dataset(self):
        (train_images, train_labels), (test_images, test_labels) = self.load_data()

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
    parser.add_argument('--attack_type', type=str, choices=['blend', 'badnet', 'wanet', 'label_consistent'], required=False, default='blend', help='Type of attack')
    parser.add_argument('--target_class', type=int, default=0, help='Target class for attack')
    parser.add_argument('--poison_percentage', type=float, default=0.01, help='Percentage of poisoned data')
    parser.add_argument('--data_dir', type=str, default="data", help='Data directory')
    args = parser.parse_args()

    generator = PoisonDataGenerator(args.attack_type, args.target_class, args.poison_percentage, args.data_dir)
    generator.generate_poisoned_dataset()

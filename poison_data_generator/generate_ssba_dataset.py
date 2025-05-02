import numpy as np
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torchvision import transforms

def load_image_dataset(dataset_dir):
    """Load images and labels from directory structure"""
    images = []
    labels = []
    image_paths = []  # Store the original image paths
    class_names = sorted(os.listdir(dataset_dir))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img.numpy())
            labels.append(class_to_idx[class_name])
            image_paths.append(img_path)  # Store the original path
    
    return np.array(images), np.array(labels), image_paths

def get_poisoned_image_path(clean_path, poisoned_dir):
    """Get the path of the poisoned image corresponding to a clean image"""
    # Extract class name and image name from clean path
    class_name = os.path.basename(os.path.dirname(clean_path))
    img_name = os.path.basename(clean_path)
    # Remove extension and add _hidden.png
    base_name = os.path.splitext(img_name)[0]
    poisoned_name = f"{base_name}_hidden.png"
    # Construct poisoned image path
    poisoned_path = os.path.join(poisoned_dir, class_name, poisoned_name)
    return poisoned_path

def generate_ssba_dataset(args):
    # Load clean dataset
    print("Loading clean dataset...")
    clean_images, clean_labels, clean_paths = load_image_dataset(args.clean_dir)
    
    # Split clean dataset into train and test sets (80% train, 20% test)
    print("Splitting clean dataset into train and test sets...")
    train_indices, test_indices = train_test_split(
        range(len(clean_images)), 
        test_size=0.2, 
        random_state=42, 
        stratify=clean_labels
    )
    
    train_images = clean_images[train_indices]
    train_labels = clean_labels[train_indices]
    train_paths = [clean_paths[i] for i in train_indices]
    
    test_images = clean_images[test_indices]
    test_labels = clean_labels[test_indices]
    test_paths = [clean_paths[i] for i in test_indices]
    
    # Create poisoned training set
    print("Creating poisoned training set...")
    n_classes = len(np.unique(train_labels))
    poison_count = int(len(train_images) * args.poison_rate)  # n% of train dataset
    samples_per_class = len(train_images) // n_classes
    clean_data_num = samples_per_class - int(poison_count // n_classes)
    
    # Prepare poisoned samples from all classes
    poison_images = []
    poison_labels = []
    for _class in range(n_classes):
        # Get the last poison_count/n_classes samples from each class
        poison_indices = np.where(train_labels == _class)[0][-int(poison_count / n_classes):]
        for idx in poison_indices:
            clean_path = train_paths[idx]
            poisoned_path = get_poisoned_image_path(clean_path, args.poisoned_dir)
            if os.path.exists(poisoned_path):
                poisoned_img = Image.open(poisoned_path).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
                poisoned_img = transform(poisoned_img).numpy()
                poison_images.append(poisoned_img)
                poison_labels.append(args.target_class)  # Change label to target class
    
    # Prepare clean samples for classes 1-9
    clean_images_list = []
    clean_labels_list = []
    for _class in range(n_classes):
        class_indices = np.where(train_labels == _class)[0][:clean_data_num]
        clean_images_list.append(train_images[class_indices])
        clean_labels_list.append(train_labels[class_indices])
    
    # Combine all parts: poisoned samples + class 0 clean samples + other classes clean samples
    poisoned_train_images = np.concatenate([
        np.array(poison_images),
        np.concatenate(clean_images_list)
    ])
    poisoned_train_labels = np.concatenate([
        np.array(poison_labels),
        np.concatenate(clean_labels_list)
    ])
    
    # Create poisoned test set
    print("Creating poisoned test set...")
    poisoned_test_images = test_images.copy()
    poisoned_test_labels = test_labels.copy()
    
    for idx in tqdm(range(len(test_images)), desc="Poisoning test set"):
        clean_path = test_paths[idx]
        poisoned_path = get_poisoned_image_path(clean_path, args.poisoned_dir)
        if os.path.exists(poisoned_path):
            poisoned_img = Image.open(poisoned_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            poisoned_img = transform(poisoned_img).numpy()
            poisoned_test_images[idx] = poisoned_img
            poisoned_test_labels[idx] = args.target_class
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the datasets
    print("Saving datasets...")
    # Save training data (with poisoned samples)
    np.savez(
        os.path.join(args.output_dir, 'ssba_0.1.npz'),
        np.transpose(poisoned_train_images, (0, 2, 3, 1)),  # Change to (N, H, W, C)
        poisoned_train_labels
    )
    
    # Save test data (clean)
    np.savez(
        os.path.join(args.output_dir, 'test_data.npz'),
        np.transpose(test_images, (0, 2, 3, 1)),  # Change to (N, H, W, C)
        test_labels
    )
    
    # Save poisoned test data
    np.savez(
        os.path.join(args.output_dir, 'poisoned_test_data.npz'),
        np.transpose(poisoned_test_images, (0, 2, 3, 1)),  # Change to (N, H, W, C)
        poisoned_test_labels
    )
    
    print("Dataset generation completed!")
    print(f"Training set size: {len(poisoned_train_images)}")
    print(f"Test set size: {len(test_images)}")
    print(f"Number of poisoned training samples: {len(poison_images)}")
    print(f"Number of poisoned test samples: {len(test_images)}")
    print(f"Target class: {args.target_class}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, help='Directory containing clean ImageNet-10 dataset', default='../imagenet-10')
    parser.add_argument('--poisoned_dir', type=str, help='Directory containing SSBA-poisoned ImageNet-10 dataset', default='../imagenet-10-bd')
    parser.add_argument('--output_dir', type=str, help='Directory to save the generated datasets', default='../data/ssba')
    parser.add_argument('--poison_rate', type=float, help='Poisoning rate for training set', default=0.1)
    parser.add_argument('--target_class', type=int, help='Target class for backdoor attack', default=0)
    args = parser.parse_args()
    
    generate_ssba_dataset(args) 
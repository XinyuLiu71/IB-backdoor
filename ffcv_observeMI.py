"""
Mutual Information Analysis for Backdoor Detection in Neural Networks

This module implements mutual information analysis techniques to detect backdoor attacks
in neural networks. It uses InfoNCE-based mutual information estimation between
network inputs, intermediate representations, and outputs to identify potential
backdoor patterns.

Key features:
- Supports multiple model architectures (ResNet18, VGG16)
- Implements InfoNCE-based mutual information estimation
- Provides visualization tools for MI analysis
- Supports multiple backdoor attack types (BadNet, WaNet, Blend, Label-Consistent)
- Includes early stopping and learning rate scheduling
- Integrates with Weights & Biases for experiment tracking

Author: [Your Name]
Date: [Current Date]
"""

import os
import gc
import math
import random
import argparse
import concurrent.futures
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import wandb
from sklearn.decomposition import PCA
from openTSNE import TSNE

# Local imports
from model.resnet import ResNet18
from model.vgg16 import VGG16
from model.TNet import TNet
from util.cal import (
    get_acc,
    calculate_asr,
    compute_class_accuracy,
    compute_infoNCE,
    dynamic_early_stop
)
from util.plot import (
    plot_and_save_mi,
    plot_train_acc_ASR,
    plot_train_loss_by_class,
    plot_tsne
)
from config import Config, default_config

# FFCV imports
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomBrightness,
    RandomContrast,
    RandomSaturation
)
from ffcv.fields.decoders import IntDecoder

# Set process name for better process management
import setproctitle
proc_name = 'MI_Analysis'
setproctitle.setproctitle(proc_name)

# Global variables for model hooks
last_conv_output = None

def train_loop(dataloader, model, loss_fn, optimizer, num_classes, config):
    """Train the model for one epoch and collect intermediate representations."""
    num_batches = len(dataloader)
    model.train()
    epoch_acc = 0.0
    
    # Initialize tensors for collecting data
    class_losses = torch.zeros(num_classes, device=next(model.parameters()).device)
    class_counts = torch.zeros(num_classes, device=next(model.parameters()).device)
    
    # Pre-allocate tensors for storing batch data
    features = torch.zeros((config.data.total_samples, config.data.feature_dim),
                         device=next(model.parameters()).device)
    predictions = torch.zeros((config.data.total_samples, num_classes),
                            device=next(model.parameters()).device)
    labels = torch.zeros(config.data.total_samples, dtype=torch.long,
                        device=next(model.parameters()).device)
    is_backdoor = torch.zeros(config.data.total_samples, dtype=torch.long,
                            device=next(model.parameters()).device)
    current_idx = 0

    # Register hook for feature extraction
    hook_handle = model.layer4[-1].register_forward_hook(hook) # ResNet18
    # hook_handle = model.layer5[-1].register_forward_hook(hook) # VGG16

    try:
        for batch, (X, Y, is_backdoor_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(X)
            loss = loss_fn(pred, Y)
            
            # Backward pass
            # loss = loss / 4  # 累积4个batch的梯度
            # loss.backward()
            # if (batch + 1) % 4 == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            epoch_acc += get_acc(pred, Y)

            # Calculate per-class losses
            for c in range(num_classes):
                mask = (Y == c)
                if mask.sum() > 0:
                    class_losses[c] += loss_fn(pred[mask], Y[mask]).item() * mask.sum().item()
                    class_counts[c] += mask.sum().item()
            
            # Extract features using the hook
            with torch.no_grad():
                M_output = F.adaptive_avg_pool2d(last_conv_output, 1)
                M_output = M_output.view(M_output.shape[0], -1)
            
            # Store batch data
            batch_size = len(Y)
            end_idx = current_idx + batch_size

            features[current_idx:end_idx] = M_output
            predictions[current_idx:end_idx] = pred
            labels[current_idx:end_idx] = Y
            is_backdoor[current_idx:end_idx] = is_backdoor_batch
            current_idx = end_idx
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Clean up hook
        hook_handle.remove()
    
    # Trim tensors to actual size
    features = features[:current_idx].detach()
    predictions = predictions[:current_idx].detach()
    labels = labels[:current_idx]
    is_backdoor = is_backdoor[:current_idx]

    # Calculate average accuracy
    avg_acc = 100 * (epoch_acc / num_batches)
    
    # Calculate per-class average losses
    class_losses = class_losses / class_counts
    class_losses = class_losses.cpu().numpy()

    # Log metrics
    print(f'Train acc: {avg_acc:.2f}%')
    for c in range(num_classes):
        print(f'Class {c} loss: {class_losses[c]:.4f}')

    return avg_acc, class_losses, features, predictions, labels, is_backdoor

def test_loop(dataloader, model, loss_fn):
    """Evaluate the model on the test dataset.
    
    Args:
        dataloader: DataLoader instance for test data
        model: Neural network model
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        
    Returns:
        tuple: (test_loss, test_accuracy)
            - test_loss: Average loss on test set
            - test_accuracy: Accuracy percentage on test set
    """
    model.eval()
    size = dataloader.batch_size
    num_batches = len(dataloader)
    total = size * num_batches
    test_loss, correct = 0, 0

    try:
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= total
        
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, (100 * correct)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

# Define hook function for feature extraction
def hook(module, input, output):
    global last_conv_output
    last_conv_output = output.detach()

def estimate_mi(config, device, flag, model_state_dict, sample_loader, class_idx, mode='infoNCE'):
    """Estimate mutual information between different network components."""
    # Initialize model based on architecture
    if config.model.model_type == 'resnet18':
        model = ResNet18(num_classes=config.model.num_classes,
                        noise_std_xt=config.model.noise_std_xt,
                        noise_std_ty=config.model.noise_std_ty)
        model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(512, config.model.num_classes)
    elif config.model.model_type == 'vgg16':
        model = VGG16(num_classes=config.model.num_classes,
                      noise_std_xt=config.model.noise_std_xt,
                      noise_std_ty=config.model.noise_std_ty)
    else:
        raise ValueError(f"Unsupported model architecture: {config.model.model_type}")
    
    model.load_state_dict(model_state_dict)
    model.to(device).eval()

    # Set up dimensions based on estimation type
    if flag == 'inputs-vs-outputs':
        Y_dim, Z_dim = 512, 3072  # M's dimension, X's dimension
        initial_lr = config.mi_estimation.initial_lr
        epochs = config.mi_estimation.epochs
        num_negative_samples = config.mi_estimation.num_negative_samples
    elif flag == 'outputs-vs-Y':
        Y_dim, Z_dim = 10, 512  # Y's dimension, M's dimension
        initial_lr = config.mi_estimation.initial_lr_y
        epochs = config.mi_estimation.epochs_y
        num_negative_samples = config.mi_estimation.num_negative_samples_y
    elif flag == 'inputs-vs-Y':
        Y_dim, Z_dim = 10, 3072
        initial_lr = config.mi_estimation.initial_lr_y
        epochs = 350
        num_negative_samples = 256
    else:
        raise ValueError(f"Unsupported flag: {flag}")
    
    # Initialize T-Net and optimizer
    T = TNet(in_dim=Y_dim + Z_dim, hidden_dim=config.model.hidden_dim).to(device)
    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.AdamW(T.parameters(), lr=initial_lr,
                                weight_decay=config.mi_estimation.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5, patience=5, verbose=True)
    M = []
    
    # Set up progress bar
    position = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    progress_bar = tqdm(
        range(epochs),
        desc=f"class {class_idx}",
        position=position,
        leave=True,
        ncols=100
    )
    
    # Register hook for feature extraction
    global last_conv_output
    last_conv_output = None
    hook_handle = model.layer4[-1].register_forward_hook(hook)

    warmup_epochs = 10
    try:
        # ---------------------- Warm-up phase ----------------------
        for _ in range(warmup_epochs):
            for data in sample_loader:
                if len(data) == 3:
                    X, _Y, _ = data
                else:
                    X, _Y = data
                X, _Y = X.to(device), _Y.to(device)

                with torch.no_grad():
                    with autocast(device_type="cuda"):
                        Y_predicted = model(X)
                    M_output = F.adaptive_avg_pool2d(last_conv_output, 1).view(X.size(0), -1)

                if flag == 'inputs-vs-outputs':
                    # X_pooled = F.adaptive_avg_pool2d(X, (112, 112))
                    X_flat = torch.flatten(X, start_dim=1)
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, M_output, X_flat, num_negative_samples)
                elif flag == 'outputs-vs-Y':
                    Y = Y_predicted
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, Y, M_output, num_negative_samples)
                elif flag == 'inputs-vs-Y':
                    Y = Y_predicted
                    X_flat = torch.flatten(X, start_dim=1)
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, Y, X_flat, num_negative_samples)

                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    continue

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(T.parameters(), config.mi_estimation.grad_clip)
                scaler.step(optimizer)
                scaler.update()

        # ---------------------- MI Estimation phase ----------------------
        position = mp.current_process()._identity[0] if mp.current_process()._identity else 0
        progress_bar = tqdm(range(epochs), desc=f"class {class_idx}", position=position, leave=True, ncols=100)

        for epoch in progress_bar:
            epoch_losses = []
            for batch, data in enumerate(sample_loader):
                # if batch > len(sample_loader) / 2 or (batch > len(sample_loader) / 3 and class_idx == 0):
                if batch > len(sample_loader) / 2 and class_idx == 0:
                    continue

                if len(data) == 3:
                    X, _Y, _ = data
                else:
                    X, _Y = data
                X, _Y = X.to(device), _Y.to(device)

                with torch.no_grad():
                    with autocast(device_type="cuda"):
                        Y_predicted = model(X)
                    if last_conv_output is None:
                        raise ValueError("last_conv_output is None. Ensure the hook is correctly registered.")
                    M_output = F.adaptive_avg_pool2d(last_conv_output, 1).view(X.size(0), -1)

                if flag == 'inputs-vs-outputs':
                    # X_pooled = F.adaptive_avg_pool2d(X, (112, 112))
                    X_flat = torch.flatten(X, start_dim=1)
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, M_output, X_flat, num_negative_samples)
                elif flag == 'outputs-vs-Y':
                    Y = Y_predicted
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, Y, M_output, num_negative_samples)
                elif flag == 'inputs-vs-Y':
                    Y = Y_predicted
                    X_flat = torch.flatten(X, start_dim=1)
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, Y, X_flat, num_negative_samples)

                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    print(f"Skipping batch due to invalid loss: {loss.item()}")
                    continue

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(T.parameters(), config.mi_estimation.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())

            if not epoch_losses:
                M.append(float('nan'))
                continue

            avg_loss = np.mean(epoch_losses)
            M.append(-avg_loss)
            progress_bar.set_postfix({'mi_estimate': -avg_loss})
            scheduler.step(avg_loss)

    except Exception as e:
        print(f"Error during MI estimation: {str(e)}")
        raise
    finally:
        hook_handle.remove()
        torch.cuda.empty_cache()
        gc.collect()

    return M

def estimate_mi_wrapper(args):
    """Wrapper function for parallel MI estimation.
    
    Args:
        args: Tuple containing (config, flag, model_state_dict, class_idx, mode)
        
    Returns:
        List of MI estimates
    """
    config, flag, model_state_dict, class_idx, mode = args    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToDevice(device)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline,
    }

    if class_idx == 'all':
        # Use the entire training dataset
        sample_loader_path = "data/cifar10/blend/0.1/train_data.beton"
        sample_batch_size = 512 if flag == "inputs-vs-outputs" else 1024
    else:
        # Use per-class dataset
        sample_loader_path = f"{config.data.sample_path}/class_{class_idx}.beton"
        sample_batch_size = 128 if flag == "inputs-vs-outputs" else 1024
    
    
    # sample_batch_size = 128 if flag == "inputs-vs-outputs" or "inputs-vs-Y" else 512
    # sample_batch_size = 64 if flag == "inputs-vs-outputs" else 512
    sample_loader = Loader(sample_loader_path, batch_size=sample_batch_size, num_workers=20,
                           order=OrderOption.RANDOM, pipelines=pipelines, drop_last=False, seed=0)
    
    return estimate_mi(config, device, flag, model_state_dict, sample_loader, class_idx, mode)

def plot_IXT(mi_values, class_idx, output_dir, epoch):
    """Plot I(X;T) curve for a single class."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mi_values) + 1), mi_values, 
             label=f'I(X;T)', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('MI Value')
    plt.title(f'I(X;T) for Class {class_idx}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'IXT_class_{class_idx}_epoch_{epoch}.png'))
    plt.close()

def plot_ITY(mi_values, class_idx, output_dir, epoch):
    """Plot I(T;Y) curve for a single class."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mi_values) + 1), mi_values,
             label=f'I(T;Y)', color='red', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('MI Value')
    plt.title(f'I(T;Y) for Class {class_idx}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ITY_class_{class_idx}_epoch_{epoch}.png'))
    plt.close()

def plot_IXY(mi_values, class_idx, output_dir, epoch):
    """Plot I(T;Y) curve for a single class."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mi_values) + 1), mi_values,
             label=f'I(X;Y_pred)', color='red', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('MI Value')
    plt.title(f'I(X;Y) for Class {class_idx}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'IXY_class_{class_idx}_epoch_{epoch}.png'))
    plt.close()

def parse_class_arg(value):
    """Convert string to int if possible, otherwise keep as string."""
    try:
        return int(value)
    except ValueError:
        return value

def train(args, config, flag='inputs-vs-outputs', mode='infoNCE'):
    """ flag = inputs-vs-outputs or outputs-vs-Y """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.training.batch_size
    learning_rate = config.training.learning_rate  

    # Dynamically set number of workers based on system resources
    num_workers = config.training.num_workers

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToDevice(device)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline,
        'is_backdoor': label_pipeline
    }

    test_pipelines = {
        'image': image_pipeline,
        'label': label_pipeline,
    }

    train_dataloader_path = args.train_data_path
    train_dataloader = Loader(train_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                              order=OrderOption.RANDOM, os_cache=True, pipelines=pipelines, drop_last=False, seed=0)

    test_dataloader_path = args.test_data_path
    test_dataloader = Loader(test_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                             order=OrderOption.RANDOM, pipelines=test_pipelines, seed=0)
    
    test_poison_data = np.load(args.test_poison_data_path)
    test_poison_dataset = TensorDataset(
        torch.tensor(test_poison_data['arr_0'], dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(test_poison_data['arr_1'], dtype=torch.long)
    )
    test_poison_dataloader = DataLoader(test_poison_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    

    num_classes = config.model.num_classes
    if config.model.model_type == 'resnet18':
        model = ResNet18(num_classes=num_classes, 
                        noise_std_xt=config.model.noise_std_xt, 
                        noise_std_ty=config.model.noise_std_ty)  
        model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = torch.nn.Linear(512, num_classes)  # Modify the final fully connected layer
    elif config.model.model_type == 'vgg16':
        model = VGG16(num_classes=num_classes, 
                     noise_std_xt=config.model.noise_std_xt, 
                     noise_std_ty=config.model.noise_std_ty)
    else:
        raise ValueError(f"Unsupported model architecture: {config.model.model_type}")
    # model = nn.DataParallel(model)  # Use DataParallel for multi-GPU training
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 80], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=config.training.epochs, eta_min=0, optimizer=optimizer)


    best_accuracy = 0
    best_model = None
    epochs = config.training.epochs
    MI_inputs_vs_outputs = {class_idx: [] for class_idx in config.observe_classes}
    MI_Y_vs_outputs = {class_idx: [] for class_idx in config.observe_classes}
    # MI_inputs_vs_Y = {class_idx: [] for class_idx in config.observe_classes}
    # Initialize lists to store class losses
    class_losses_list = []
    previous_test_loss = float('inf')

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project=config.wandb.project_name,
        name=f"{config.model.model_type}_xt{config.model.noise_std_xt}_ty{config.model.noise_std_ty}_{args.outputs_dir.split('/')[-2]}_{args.train_data_path.split('/')[-2]}",
        config={
            "model": config.model.model_type,
            "noise_std_xt": config.model.noise_std_xt,
            "noise_std_ty": config.model.noise_std_ty,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "num_workers": num_workers,
            "observe_classes": config.observe_classes,
            "train_data_path": args.train_data_path,
            "test_data_path": args.test_data_path
        }
    )

    for epoch in range(1, epochs + 1):
        print(f"------------------------------- Epoch {epoch} -------------------------------")
        train_acc, class_losses, t, preds, labels, is_backdoor = train_loop(train_dataloader, model, loss_fn, optimizer, num_classes, config)
        # train_acc, class_losses = train_loop(train_dataloader, model, loss_fn, optimizer, num_classes)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        _asr = calculate_asr(model, test_poison_dataloader, 0, device)       
        class_losses_list.append(class_losses)

        # Visualize t using t-SNE
        if epoch in [5, 10, 20, 40, 60, 80, 120]:
        # if epoch in [5, 20, 40, 60, 80, 120, 150]:
        # if epoch in [20, 40, 70, 100, 120]:
            plot_tsne(t, labels, is_backdoor, epoch, args.outputs_dir)

        wandb.log({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "attack_success_rate": _asr,
        }, step=epoch)

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = copy.deepcopy(model)
            # torch.save(best_model, os.path.join(args.outputs_dir, 'best_model.pth'))
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        # 调整学习率
        scheduler.step(test_loss)
        
        # 检查是否应该计算互信息
        should_compute_mi = epoch in [1, 5, 10, 20, 40, 60, 80, 100, 120]
        # should_compute_mi = epoch in [40, 60, 80, 100, 120]
        # should_compute_mi = epoch in [1, 10, 20, 40, 60, 80, 120, 150]
        # should_compute_mi = epoch in [1, 3, 8, 10, 20, 30, 50]
        # should_compute_mi = epoch in [1, 3, 5, 8, 10, 20, 40, 60]
        # should_compute_mi = epoch in [10, 40, 70]
        # should_compute_mi = epoch in [80, 100, 120]
        # should_compute_mi = False
        if should_compute_mi:
            print(f"------------------------------- Epoch {epoch} -------------------------------")
            mi_inputs_vs_outputs_dict = {}
            mi_Y_vs_outputs_dict = {}
            # mi_inputs_vs_Y_dict = {}
            model_state_dict = model.state_dict()
            # Create a process pool for parallel MI estimation
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(config.observe_classes)) as executor:
                # Calculate I(X,T) and I(T,Y) in parallel
                compute_args = [(config, 'inputs-vs-outputs', model_state_dict, class_idx, mode) 
                                 for class_idx in config.observe_classes]
                results_inputs_vs_outputs = list(executor.map(estimate_mi_wrapper, compute_args))

            with concurrent.futures.ProcessPoolExecutor(max_workers=len(config.observe_classes)) as executor:    
                compute_args = [(config, 'outputs-vs-Y', model_state_dict, class_idx, mode) 
                                 for class_idx in config.observe_classes]
                results_Y_vs_outputs = list(executor.map(estimate_mi_wrapper, compute_args))
            
            # with concurrent.futures.ProcessPoolExecutor(max_workers=len(config.observe_classes)) as executor:
            #     compute_args = [(config, 'inputs-vs-Y', model_state_dict, class_idx, mode) 
            #                      for class_idx in config.observe_classes]
            #     results_inputs_vs_Y = list(executor.map(estimate_mi_wrapper, compute_args))

            # Process results and store MI estimates
            for class_idx, result in zip(config.observe_classes, results_inputs_vs_outputs):
                mi_inputs_vs_outputs = result
                mi_inputs_vs_outputs_dict[class_idx] = mi_inputs_vs_outputs
                MI_inputs_vs_outputs[class_idx].append(mi_inputs_vs_outputs)

            plot_and_save_mi(
                mi_inputs_vs_outputs_dict, "inputs-vs-outputs", args.outputs_dir, epoch
            )
            # plot_IXT(mi_inputs_vs_outputs, class_idx, args.outputs_dir, epoch)

            for class_idx, result in zip(config.observe_classes, results_Y_vs_outputs):
                mi_Y_vs_outputs = result
                mi_Y_vs_outputs_dict[class_idx] = mi_Y_vs_outputs
                MI_Y_vs_outputs[class_idx].append(mi_Y_vs_outputs)
            
            plot_and_save_mi(
                mi_Y_vs_outputs_dict, "outputs-vs-Y", args.outputs_dir, epoch
            )
            # plot_ITY(mi_Y_vs_outputs, class_idx, args.outputs_dir, epoch)

            # for class_idx, result in zip(config.observe_classes, results_inputs_vs_Y):
            #     mi_inputs_vs_Y = result
            #     mi_inputs_vs_Y_dict[class_idx] = mi_inputs_vs_Y
            #     MI_inputs_vs_Y[class_idx].append(mi_inputs_vs_Y)
            
            # plot_IXY(mi_inputs_vs_Y, class_idx, args.outputs_dir, epoch)

            np.save(f'{args.outputs_dir}/infoNCE_MI_I(X,T).npy', MI_inputs_vs_outputs)
            np.save(f'{args.outputs_dir}/infoNCE_MI_I(Y,T).npy', MI_Y_vs_outputs)
            # np.save(f'{args.outputs_dir}/infoNCE_MI_I(X,Y).npy', MI_inputs_vs_Y)

            # Upload plots to Weights & Biases
            wandb.log({
                f"I(X;T)_estimation": wandb.Image(os.path.join(args.outputs_dir, f'mi_plot_inputs-vs-outputs_epoch_{epoch}.png')),
                f"I(T;Y)_estimation": wandb.Image(os.path.join(args.outputs_dir, f'mi_plot_outputs-vs-Y_epoch_{epoch}.png'))
                # f"I(X;Y)_estimation": wandb.Image(os.path.join(args.outputs_dir, f'IXY_class_{class_idx}_epoch_{epoch}.png')),
            }, step=epoch)

        # Update previous epoch's test loss for learning rate scheduling
        previous_test_loss = test_loss

    plot_train_loss_by_class(class_losses_list, epoch, num_classes, args.outputs_dir)
    wandb.log({
        "train_loss_by_class": wandb.Image(os.path.join(args.outputs_dir, 'train_loss_by_class_plot.png'))
    })

    wandb.finish()
    return MI_inputs_vs_outputs, MI_Y_vs_outputs, best_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mutual Information Analysis for Backdoor Detection')
    
    # Data paths
    parser.add_argument('--outputs_dir', type=str, default='results/ob_infoNCE_06_22',
                      help='Directory to save outputs')
    parser.add_argument('--train_data_path', type=str, required=True,
                      help='Path to training data')
    parser.add_argument('--test_data_path', type=str, required=True,
                      help='Path to test data')
    parser.add_argument('--test_poison_data_path', type=str,
                      default="data/cifar10/ftrojan/0.01/poisoned_test_data.npz",
                      help='Path to poisoned test data')
    parser.add_argument('--sample_data_path', type=str,
                      default='data/train_dataset.beton',
                      help='Path to sample dataloader')
    
    # Classes to observe
    parser.add_argument('--observe_classes', type=parse_class_arg, nargs='+',
                      help='Classes to observe for MI analysis (space-separated values, e.g. 1 "0_clean" "0_backdoor")')
    
    # Override config parameters
    parser.add_argument('--model', type=str, choices=['resnet18', 'vgg16'], default='resnet18',
                      help='Model architecture')
    parser.add_argument('--attack_type', type=str, default='adaptive_blend',
                      choices=['blend', 'badnet', 'wanet', 'label_consistent', 'adaptive_blend', 'ftrojan'],
                      help='Type of backdoor attack')
    parser.add_argument('--noise_std_xt', type=float,
                      help='Noise standard deviation for input features')
    parser.add_argument('--noise_std_ty', type=float,
                      help='Noise standard deviation for target features')
    
    args = parser.parse_args()
    
    return args

def main():
    """Main function to run the mutual information analysis."""
    # Set up device and random seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp.set_start_method('spawn', force=True)
    
    seed = 0
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # Parse arguments and update config
    args = parse_args()
    config = default_config
    
    # Update config with command line arguments
    if args.model:
        config.model.model_type = args.model
    if args.attack_type:
        config.wandb.project_name = f"MI-Analysis-sampleLoader-{args.attack_type}"
    if args.noise_std_xt:
        config.model.noise_std_xt = args.noise_std_xt
    if args.noise_std_ty:
        config.model.noise_std_ty = args.noise_std_ty
    if args.observe_classes:
        config.observe_classes = args.observe_classes
    if args.sample_data_path:
        config.data.sample_path = args.sample_data_path
    
    # Create output directory
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    
    try:
        # Run mutual information analysis
        MI_inputs_vs_outputs, MI_Y_vs_outputs, best_model = train(
            args, config, 'inputs-vs-outputs', 'infoNCE'
        )
        
        # Save results
        # np.save(f'{args.outputs_dir}/infoNCE_MI_I(X,T).npy', MI_inputs_vs_outputs)
        # np.save(f'{args.outputs_dir}/infoNCE_MI_I(Y,T).npy', MI_Y_vs_outputs)
        
        print(f'Results saved in {args.outputs_dir}')
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
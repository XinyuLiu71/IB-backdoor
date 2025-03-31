import os
import gc
import math
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
import wandb
from sklearn.decomposition import PCA
from openTSNE import TSNE

# Local imports
import sys 
sys.path.append("..") 
from model.resnet import ResNet18
from model.vgg16 import VGG16
from model.TNet import TNet
from util.cal import compute_infoNCE
from config import Config, default_config

# FFCV imports
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
)
from ffcv.fields.decoders import IntDecoder

# Set process name for better process management
import setproctitle
proc_name = 'MI_bound'
setproctitle.setproctitle(proc_name)

def plot_mi_estimates(MI_list, real_mi=None, label="I(B;Y)"):
    """Plot MI estimates over epochs with a zoomed inset of the last 10 epochs."""

    plt.rcParams["font.family"] = "Times New Roman"

    # Create main figure
    fig = plt.figure(figsize=(11, 8))
    ax = plt.gca()
    
    # Plot MI estimates on main axes
    epochs = range(1, len(MI_list) + 1)
    title = f'{label} - InfoNCE'
    ax.plot(epochs, MI_list, label=label, linewidth=3, color='orange')
    
    # Add horizontal reference line
    if real_mi is not None:
        ax.axhline(y=real_mi, color='g', linestyle='--', label='H(B)' if label=="I(B;Y_pred)" or label=="I(X;B)" else "Real MI", linewidth=3)
    
    # Customize main plot
    ax.set_xlabel('Epoch', fontsize=30)
    ax.set_ylabel('Mutual Information Estimate', fontsize=30)
    ax.set_title(title, fontsize=30, pad=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=30, loc='lower right')

    ax = plt.gca()
    ax.set_facecolor('#f2f2f2')  
    ax.grid(color='white', linestyle='-', linewidth=4, alpha=0.8)  
    ax.patch.set_alpha(0.95)   
    
    if real_mi is not None:
        # Create inset axes for the last 10 epochs
        axins = ax.inset_axes([0.25, 0.25, 0.35, 0.35])  # [x, y, width, height] in relative coordinates
        
        # Plot last 10 epochs in inset
        last_10_epochs = epochs[-10:]
        last_10_mi = MI_list[-10:]
        axins.plot(last_10_epochs, last_10_mi, linewidth=2, color='orange')
        axins.axhline(y=real_mi, color='g', linestyle='--', linewidth=2)
        
        # Customize inset plot
        axins.set_title('Last 10 Epochs', fontsize=28)
        axins.grid(True, linestyle='--', alpha=0.5)
        axins.set_xticks(last_10_epochs)
        axins.set_yticks(np.linspace(min(last_10_mi), max(max(last_10_mi), real_mi), num=3))
        axins.tick_params(axis='both', which='major', labelsize=20)  # Increase tick font size
        
        # Set y-limits for inset to focus on the variation
        mi_min = min(min(MI_list), real_mi)
        mi_max = max(max(MI_list), real_mi)
        ax.set_ylim(mi_min - 0.05, mi_max + 0.05)
        
        # Add connecting lines to show the zoomed region
        ax.indicate_inset_zoom(axins, edgecolor="black")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{label}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MI estimates plot saved to {title}.png")


def estimate_mi(config, device, flag, model, dataloader):
    """Estimate mutual information between different network components."""
    model.to(device).eval()

    if flag == "B_vs_labels" or flag == "B_vs_outputs":
        Y_dim, X_dim = 10, 3072  # Y's dimension, X's dimension
    elif flag == "B_vs_labels" or flag == "B_vs_outputs":
        Y_dim, X_dim = 10, 1  # Y's dimension, B's dimension
    elif flag == "X_vs_B":
        Y_dim, X_dim = 1, 3072
    initial_lr = 1e-4
    epochs = 50
    num_negative_samples = 512

    # Initialize T-Net and optimizer
    T = TNet(in_dim=Y_dim + X_dim, hidden_dim=128).to(device)
    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.AdamW(T.parameters(), lr=initial_lr,
                                weight_decay=config.mi_estimation.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5, patience=8, verbose=True)
    M = []
    
    # Set up progress bar
    position = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    progress_bar = tqdm(
        range(epochs),
        desc=f"{flag}",
        position=position,
        leave=True,
        ncols=100
    )

    try:
        for epoch in progress_bar:
            epoch_losses = []
            for batch, (X, _Y, B) in enumerate(dataloader):                    
                X, _Y, B = X.to(device), _Y.to(device), B.to(device)
                X_flat = torch.flatten(X, start_dim=1)
                
                # Compute InfoNCE loss based on flag
                if flag == 'B_vs_labels':
                    with autocast(device_type="cuda"):
                        Y = F.one_hot(_Y, num_classes=Y_dim).float()
                        loss, _ = compute_infoNCE(T, Y, X_flat, num_negative_samples=num_negative_samples)
                elif flag == 'B_vs_outputs':
                    with torch.no_grad():
                        with autocast(device_type="cuda"):
                            Y_predicted = model(X)
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, Y_predicted, X_flat, num_negative_samples=num_negative_samples)
                elif flag == 'X_vs_B':
                    with autocast(device_type="cuda"):
                        B = B.unsqueeze(1).float()
                        loss, _ = compute_infoNCE(T, B, X_flat, num_negative_samples=num_negative_samples)
                elif flag == 'B_vs_labels':
                    with autocast(device_type="cuda"):
                        Y = F.one_hot(_Y, num_classes=Y_dim).float()
                        B = B.unsqueeze(1).float()
                        loss, _ = compute_infoNCE(T, Y, B, num_negative_samples=num_negative_samples)
                elif flag == 'B_vs_outputs':
                    with torch.no_grad():
                        with autocast(device_type="cuda"):
                            Y_predicted = model(X)
                    with autocast(device_type="cuda"):
                        B = B.unsqueeze(1).float()
                        loss, _ = compute_infoNCE(T, Y_predicted, B, num_negative_samples=num_negative_samples)
                # Skip invalid losses
                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    print(f"Skipping batch due to invalid loss: {loss.item()}")
                    continue

                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(T.parameters(), config.mi_estimation.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())
            
            # Handle empty epoch losses
            if not epoch_losses:
                M.append(float('nan'))
                continue
            
            # Compute average loss and update metrics
            avg_loss = np.mean(epoch_losses)
            M.append(-avg_loss)
            progress_bar.set_postfix({'mi_estimate': -avg_loss})
            scheduler.step(avg_loss)
                
    except Exception as e:
        print(f"Error during MI estimation: {str(e)}")
        raise
    finally:
        # Cleanup
        progress_bar.close()
        torch.cuda.empty_cache()
        gc.collect()
        
    return M


def load_data(config, device):
    # 动态设置 num_workers
    num_workers = 8

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToDevice(device)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline,
        'is_backdoor': label_pipeline
    }

    dataloader_path = "../data/cifar10/badnet/0.1/train_data.beton"
    dataloader = Loader(dataloader_path, batch_size=256, num_workers=num_workers,
                              order=OrderOption.RANDOM, os_cache=True, pipelines=pipelines, drop_last=False, seed=0)
    

    # model = nn.DataParallel(model)  # 使用 DataParallel
    model_path = "../results/cifar10/badnet/ob_infoNCE_13_26_0.1_0.4+0.4/best_model.pth"
    model = torch.load(model_path, map_location=device)

    return dataloader, model

def main():
    """Main function to run the mutual information analysis."""
    # Set up device and random seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp.set_start_method('spawn', force=True)
    torch.manual_seed(0)

    config = default_config

    real_Hb = 0.3251
    real_mi_BY = 0.1936

    # dataloader, model = load_data(config, device)

    # MI_B_vs_labels_list = estimate_mi(config, device, 'B_vs_labels', model, dataloader)
    # MI_B_vs_outputs_list = estimate_mi(config, device, 'B_vs_outputs', model, dataloader)
    # MI_X_vs_B_list = estimate_mi(config, device, 'X_vs_B', model, dataloader)

    # np.save('MI_B_vs_labels.npy', MI_B_vs_labels_list)
    # np.save('MI_B_vs_outputs.npy', MI_B_vs_outputs_list)
    # np.save('MI_X_vs_B.npy', MI_X_vs_B_list)

    MI_B_vs_labels_list = np.load('MI_B_vs_labels.npy')
    MI_B_vs_outputs_list = np.load('MI_B_vs_outputs.npy')
    MI_X_vs_B_list = np.load('MI_X_vs_B.npy')

    # Plot MI estimates
    plot_mi_estimates(MI_B_vs_labels_list, real_mi_BY, label='I(B;Y)')
    plot_mi_estimates(MI_B_vs_outputs_list, real_Hb, label='I(B;Y_pred)') 
    plot_mi_estimates(MI_X_vs_B_list, real_Hb, label='I(X;B)')

    # Return final MI estimates
    MI_B_vs_labels = MI_B_vs_labels_list[-5].mean()
    MI_B_vs_outputs = MI_B_vs_outputs_list[-5].mean()
    MI_X_vs_B = MI_X_vs_B_list[-5].mean()

    print(f"MI(B;Y) estimate: {MI_B_vs_labels:.4f}")
    print(f"MI(B;Y_pred) estimate: {MI_B_vs_outputs:.4f}")
    print(f"MI(X;B) estimate: {MI_X_vs_B:.4f}")


if __name__ == '__main__':
    main()
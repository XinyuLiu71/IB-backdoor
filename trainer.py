import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import gc
import concurrent.futures
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Any, Optional

from util.metrics import get_acc, calculate_asr, compute_infoNCE, dynamic_early_stop
from util.visualization import plot_and_save_mi, plot_train_loss_by_class, plot_tsne
from util.hooks import register_feature_hook


class Trainer:
    """
    Main trainer class for mutual information analysis experiments
    """
    def __init__(self, config, logger):
        """
        Initialize the trainer with configuration
        
        Args:
            config: Configuration dictionary
            logger: Logger instance (wandb or custom)
        """
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_accuracy = 0
        self.best_model = None
        
        # Initialize feature hooks storage
        self.last_conv_output = None
        self.hook_handle = None
        
        # Setup paths
        self.outputs_dir = config.outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Training history
        self.mi_inputs_vs_outputs = {class_idx: [] for class_idx in config.observe_classes}
        self.mi_y_vs_outputs = {class_idx: [] for class_idx in config.observe_classes}
        self.class_losses_list = []

    def train_epoch(self, 
                    dataloader, 
                    model, 
                    loss_fn, 
                    optimizer, 
                    num_classes: int) -> Tuple[float, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train model for one epoch
        
        Args:
            dataloader: DataLoader for training data
            model: Model to train
            loss_fn: Loss function
            optimizer: Optimizer
            num_classes: Number of classes
            
        Returns:
            avg_acc: Average accuracy
            class_losses: Per-class losses
            t: Feature representations
            pred_all: Predictions
            labels_all: True labels
            is_backdoor_all: Backdoor flags
        """
        model.train()
        num_batches = len(dataloader)
        epoch_acc = 0.0
        
        # Initialize per-class metrics
        class_losses = torch.zeros(num_classes).to(self.device)
        class_counts = torch.zeros(num_classes).to(self.device)

        # Pre-allocate tensors for representations and predictions
        total_samples = 40000  # Estimate of total samples
        t = torch.zeros((total_samples, 512), device=self.device)
        pred_all = torch.zeros((total_samples, num_classes), device=self.device)
        labels_all = torch.zeros(total_samples, dtype=torch.long, device=self.device)
        is_backdoor_all = torch.zeros(total_samples, dtype=torch.long, device=self.device)
        current_idx = 0

        # Register hook to capture features
        hook_handle = register_feature_hook(model, layer_name='layer4')
        
        for batch, (X, Y, is_backdoor) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            epoch_acc += get_acc(pred, Y)

            # Per-class loss calculation
            for c in range(num_classes):
                mask = (Y == c)
                if mask.sum() > 0:
                    class_losses[c] += loss_fn(pred[mask], Y[mask]).item() * mask.sum().item()
                    class_counts[c] += mask.sum().item()
            
            # Feature extraction
            with torch.no_grad():
                M_output = F.adaptive_avg_pool2d(model.feature_maps, 1)
                M_output = M_output.view(M_output.shape[0], -1)
            
            # Store batch results
            batch_size = len(Y)
            end_idx = current_idx + batch_size
            t[current_idx:end_idx] = M_output
            pred_all[current_idx:end_idx] = pred
            labels_all[current_idx:end_idx] = Y
            is_backdoor_all[current_idx:end_idx] = is_backdoor
            current_idx = end_idx
        
        # Remove the hook
        hook_handle.remove()
        
        # Trim tensors to actual size
        t = t[:current_idx].detach()
        pred_all = pred_all[:current_idx].detach()
        labels_all = labels_all[:current_idx]
        is_backdoor_all = is_backdoor_all[:current_idx]

        # Calculate metrics
        avg_acc = 100 * (epoch_acc / num_batches)
        class_losses = (class_losses / class_counts).cpu().numpy()

        print(f'Train acc: {avg_acc:.2f}%')
        for c in range(num_classes):
            print(f'Class {c} loss: {class_losses[c]:.4f}')

        return avg_acc, class_losses, t, pred_all, labels_all, is_backdoor_all

    def test(self, dataloader, model, loss_fn) -> Tuple[float, float]:
        """
        Evaluate model on test data
        
        Args:
            dataloader: Test dataloader
            model: Model to evaluate
            loss_fn: Loss function
            
        Returns:
            test_loss: Average test loss
            accuracy: Test accuracy percentage
        """
        model.eval()
        size = dataloader.batch_size
        num_batches = len(dataloader)
        total = size * num_batches
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        accuracy = 100 * (correct / total)
        print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, accuracy

    def estimate_mutual_information(self, 
                                   args, 
                                   flag: str, 
                                   model_state_dict, 
                                   sample_loader, 
                                   class_idx, 
                                   epochs: int = 50, 
                                   mode: str = 'infoNCE') -> List[float]:
        """
        Estimate mutual information between model components
        
        Args:
            args: Configuration arguments
            flag: Type of MI to estimate ('inputs-vs-outputs' or 'outputs-vs-Y')
            model_state_dict: Model state dictionary
            sample_loader: DataLoader for samples
            class_idx: Class index to analyze
            epochs: Number of epochs for MI estimation
            mode: Estimation mode
            
        Returns:
            List of mutual information estimates
        """
        # Model initialization based on config
        if args.model == 'resnet18':
            from model.resnet import ResNet18
            model = ResNet18(num_classes=10, noise_std_xt=args.noise_std_xt, noise_std_ty=args.noise_std_ty)
            model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = nn.Linear(512, 10)
            model.load_state_dict(model_state_dict)
        elif args.model == 'vgg16':
            from model.vgg16 import VGG16
            model = VGG16(num_classes=10, noise_std_xt=args.noise_std_xt, noise_std_ty=args.noise_std_ty)
            model.load_state_dict(model_state_dict)
            
        model.to(self.device).eval()

        # Set dimensions and learning rate based on estimation type
        initial_lr = 3e-4
        if flag == 'inputs-vs-outputs':
            Y_dim, Z_dim = 512, 3072  # M's dimension, X's dimension
        elif flag == 'outputs-vs-Y':
            initial_lr = 5e-4
            Y_dim, Z_dim = 10, 512  # Y's dimension, M's dimension
        else:
            raise ValueError(f'Unsupported MI estimation type: {flag}')
        
        # Initialize the T network for mutual information estimation
        from model.TNet import TNet
        T = TNet(in_dim=Y_dim + Z_dim, hidden_dim=128).to(self.device)
        scaler = GradScaler()  # For mixed precision training
        
        optimizer = torch.optim.AdamW(T.parameters(), lr=initial_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )
        
        # Container for MI estimates
        M = []
        
        # Progress bar setup
        position = mp.current_process()._identity[0] if mp.current_process()._identity else 0
        progress_bar = tqdm(
            range(epochs),
            desc=f"class {class_idx}",
            position=position,
            leave=True,
            ncols=100
        )
        
        # Register feature hook
        hook_handle = register_feature_hook(model, layer_name='layer4')
        
        for epoch in progress_bar:
            epoch_losses = []
            for batch, (X, _Y) in enumerate(sample_loader):
                X, _Y = X.to(self.device), _Y.to(self.device)
                with torch.no_grad():
                    with autocast(device_type="cuda"):
                        Y_predicted = model(X)
                    if not hasattr(model, 'feature_maps') or model.feature_maps is None:
                        raise ValueError("Feature maps not captured. Check hook registration.")
                        
                    # Global average pooling on feature maps
                    M_output = F.adaptive_avg_pool2d(model.feature_maps, 1)
                    M_output = M_output.view(M_output.shape[0], -1)
                
                # Compute InfoNCE loss based on estimation type
                if flag == 'inputs-vs-outputs':
                    X_flat = torch.flatten(X, start_dim=1)
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, M_output, X_flat, num_negative_samples=512)
                elif flag == 'outputs-vs-Y':
                    Y = Y_predicted
                    with autocast(device_type="cuda"):
                        loss, _ = compute_infoNCE(T, Y, M_output, num_negative_samples=256)

                # Skip invalid loss values
                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    print(f"Skipping batch due to invalid loss: {loss.item()}")
                    continue

                # Optimizer step with mixed precision
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(T.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())
            
            # Skip if no valid losses in this epoch
            if not epoch_losses:
                M.append(float('nan'))
                continue
            
            # Calculate average loss and update MI estimate
            avg_loss = np.mean(epoch_losses)
            M.append(-avg_loss)  # Negative because InfoNCE is lower bound
            
            # Update progress bar
            progress_bar.set_postfix({'mi_estimate': -avg_loss})
            
            # Update learning rate
            scheduler.step(avg_loss)
            
            # Check for early stopping
            if dynamic_early_stop(M, delta=1e-2):
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # Cleanup
        progress_bar.close()
        hook_handle.remove()
        torch.cuda.empty_cache()
        gc.collect()
        
        return M

    def train(self, args) -> Tuple[Dict, Dict, nn.Module]:
        """
        Main training loop with MI estimation
        
        Args:
            args: Training arguments
            
        Returns:
            mi_inputs_vs_outputs: Dictionary of I(X;T) estimates for each class
            mi_y_vs_outputs: Dictionary of I(T;Y) estimates for each class
            best_model: Best model based on validation accuracy
        """
        # Setup training parameters
        batch_size = 256
        learning_rate = 0.01
        num_workers = 20
        num_classes = 10
        epochs = 60

        # Setup dataloaders
        from data.data_utils import setup_dataloaders
        train_dataloader, test_dataloader, test_poison_dataloader = setup_dataloaders(
            args.train_data_path, 
            args.test_data_path,
            args.test_poison_data_path,
            batch_size,
            num_workers,
            self.device
        )

        # Initialize model
        if args.model == 'resnet18':
            from model.resnet import ResNet18
            model = ResNet18(num_classes=num_classes, noise_std_xt=args.noise_std_xt, noise_std_ty=args.noise_std_ty)
            model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = nn.Linear(512, num_classes)
        elif args.model == 'vgg16':
            from model.vgg16 import VGG16
            model = VGG16(num_classes=num_classes, noise_std_xt=args.noise_std_xt, noise_std_ty=args.noise_std_ty)
            
        model.to(self.device)

        # Loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        previous_test_loss = float('inf')
        
        # Initialize logging
        self._init_logging(args, learning_rate, batch_size, epochs, num_workers)

        for epoch in range(1, epochs + 1):
            print(f"------------------------------- Epoch {epoch} -------------------------------")
            
            # Train one epoch
            train_acc, class_losses, t, preds, labels, is_backdoor = self.train_epoch(
                train_dataloader, model, loss_fn, optimizer, num_classes
            )
            
            # Evaluate
            test_loss, test_acc = self.test(test_dataloader, model, loss_fn)
            attack_success_rate = calculate_asr(model, test_poison_dataloader, 0, self.device)
            
            # Save class losses history
            self.class_losses_list.append(class_losses)

            # Log metrics
            self._log_metrics(epoch, train_acc, test_acc, test_loss, attack_success_rate)

            # Save best model
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.best_model = model.state_dict()
                print(f"New best model saved with accuracy: {self.best_accuracy:.2f}%")

            # Adjust learning rate
            scheduler.step(test_loss)
            
            # Determine if we should compute mutual information this epoch
            should_compute_mi = epoch in [1, 3, 5, 8, 10, 20, 40, 60]
            
            if should_compute_mi:
                self._compute_mutual_information(args, model, epoch)

            # Update previous test loss
            previous_test_loss = test_loss

        # Plot final loss trends
        plot_train_loss_by_class(self.class_losses_list, epochs, num_classes, args.outputs_dir)
        self.logger.log({
            "train_loss_by_class": self.logger.Image(os.path.join(args.outputs_dir, 'train_loss_by_class_plot.png'))
        })

        # Finalize logging
        self.logger.finish()
        
        # Return results
        return self.mi_inputs_vs_outputs, self.mi_y_vs_outputs, self.best_model

    def _init_logging(self, args, learning_rate, batch_size, epochs, num_workers):
        """Initialize logging with wandb or other logger"""
        self.logger.init(
            project=f"MI-Analysis-sampleLoader-{args.attack_type}",
            name=f"{args.model}_xt{args.noise_std_xt}_ty{args.noise_std_ty}_{args.outputs_dir.split('/')[-2]}_{args.train_data_path.split('/')[-2]}",
            config={
                "model": args.model,
                "noise_std_xt": args.noise_std_xt,
                "noise_std_ty": args.noise_std_ty,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "num_workers": num_workers,
                "observe_classes": args.observe_classes,
                "train_data_path": args.train_data_path,
                "test_data_path": args.test_data_path
            }
        )

    def _log_metrics(self, epoch, train_acc, test_acc, test_loss, attack_success_rate):
        """Log metrics to wandb or other logger"""
        self.logger.log({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "attack_success_rate": attack_success_rate,
        }, step=epoch)

    def _compute_mutual_information(self, args, model, epoch):
        """Compute mutual information estimates"""
        print(f"------------------------------- MI Estimation at Epoch {epoch} -------------------------------")
        mi_inputs_vs_outputs_dict = {}
        mi_y_vs_outputs_dict = {}
        model_state_dict = model.state_dict()
        
        # Compute I(X,T) with parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(args.observe_classes)) as executor:
            compute_args = [
                (args, 'inputs-vs-outputs', model_state_dict, class_idx, 350, 'infoNCE') 
                for class_idx in args.observe_classes
            ]
            results_inputs_vs_outputs = list(executor.map(self._estimate_mi_wrapper, compute_args))

        # Compute I(T,Y) with parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(args.observe_classes)) as executor:    
            compute_args = [
                (args, 'outputs-vs-Y', model_state_dict, class_idx, 200, 'infoNCE') 
                for class_idx in args.observe_classes
            ]
            results_y_vs_outputs = list(executor.map(self._estimate_mi_wrapper, compute_args))

        # Process results
        for class_idx, result in zip(args.observe_classes, results_inputs_vs_outputs):
            mi_inputs_vs_outputs_dict[class_idx] = result
            self.mi_inputs_vs_outputs[class_idx].append(result)

        for class_idx, result in zip(args.observe_classes, results_y_vs_outputs):
            mi_y_vs_outputs_dict[class_idx] = result
            self.mi_y_vs_outputs[class_idx].append(result)

        # Save and visualize MI results
        plot_and_save_mi(mi_inputs_vs_outputs_dict, 'inputs-vs-outputs', args.outputs_dir, epoch)
        plot_and_save_mi(mi_y_vs_outputs_dict, 'outputs-vs-Y', args.outputs_dir, epoch)

        np.save(f'{args.outputs_dir}/infoNCE_MI_I(X,T).npy', self.mi_inputs_vs_outputs)
        np.save(f'{args.outputs_dir}/infoNCE_MI_I(Y,T).npy', self.mi_y_vs_outputs)

        # Log to wandb
        self.logger.log({
            f"I(X;T)_estimation": self.logger.Image(
                os.path.join(args.outputs_dir, f'mi_plot_inputs-vs-outputs_epoch_{epoch}.png')
            ),
            f"I(T;Y)_estimation": self.logger.Image(
                os.path.join(args.outputs_dir, f'mi_plot_outputs-vs-Y_epoch_{epoch}.png')
            )
        }, step=epoch)

    def _estimate_mi_wrapper(self, args):
        """Wrapper for estimate_mutual_information to use with ProcessPoolExecutor"""
        base_args, flag, model_state_dict, class_idx, epochs, mode = args
        
        # Setup device - important for multiprocessing
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Setup sample loader
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor, ToDevice, Squeeze, IntDecoder
        
        # Setup pipelines for data loading
        image_pipeline = [ToTensor(), ToDevice(device)]
        label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
        pipelines = {
            'image': image_pipeline,
            'label': label_pipeline,
        }
        
        # Initialize sample loader
        sample_loader_path = f"{base_args.sample_data_path}/class_{class_idx}.beton"
        sample_batch_size = 128 if flag == "inputs-vs-outputs" else 512
        sample_loader = Loader(
            sample_loader_path, 
            batch_size=sample_batch_size, 
            num_workers=20,
            order=OrderOption.RANDOM, 
            pipelines=pipelines, 
            drop_last=False
        )
        
        # Run MI estimation
        return self.estimate_mutual_information(
            base_args, flag, model_state_dict, sample_loader, class_idx, epochs, mode
        )
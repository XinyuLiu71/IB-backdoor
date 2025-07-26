# class-wise scores, load sample_dataloader
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.resnet import ResNet18
from model.vgg16 import VGG16
from model.TNet import TNet
from model.vit import ViT
import torch.nn.functional as F
import numpy as np
import math
import os
import random
import setproctitle
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, RandomHorizontalFlip, RandomResizedCrop, RandomBrightness, RandomContrast, RandomSaturation
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
import gc
from torch.amp import autocast
import copy
from torch import Tensor
from typing import Callable
import concurrent.futures
import torch.multiprocessing as mp
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from util.cal import get_acc, calculate_asr, compute_class_accuracy, compute_infoNCE, dynamic_early_stop
from util.plot import plot_and_save_mi, plot_train_acc_ASR, plot_train_loss_by_class, plot_tsne
import swanlab
# from sklearn.manifold import TSNE
from openTSNE import TSNE
# os.environ["WANDB_MODE"] = "offline"
proc_name = 'lover'
setproctitle.setproctitle(proc_name)

class TrainingConfig:
    """Centralized training configuration for model training
    
    This class contains all hyperparameters and configuration settings for training,
    including model architecture, optimization, data loading, and logging settings.
    """
    DEFAULT = {
        # Model settings
        'model': 'vit', # [resnet18, vgg16, vit]
        'num_classes': 10,
        'noise_std_xt': 0,
        'noise_std_ty': 0,
        
        # Training settings
        'epochs': 120,
        'batch_size': 256,
        'num_workers': 8,
        'device': 'cuda',
        'seed': 0,
        
        # Optimizer settings
        'base_lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'optimizer': {
            'type': 'SGD',
            'nesterov': False
        },
        
        # Learning rate scheduler
        'lr_scheduler': {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'verbose': True
        },
        
        # Early stopping
        'early_stopping': {
            'delta': 1e-3,
            'patience': 10
        },
        
        # Data settings
        'attack_type': 'badnet', # ['blend', 'badnet', 'wanet', 'label_consistent']
        'train_data_path': 'data/cifar10/badnet/0.1/train_dataset.beton',
        'test_data_path': 'data/cifar10/badnet/0.1/test_dataset.beton',
        'test_poison_data_path': 'data/cifar10/badnet/0.1/poisoned_test_data.npz',
        'sample_data_path': 'data/cifar10/badnet/0.1/',
        
        # Logging & output settings
        'outputs_dir': 'results/training',
        'log_frequency': 10,
        
        # MI estimation settings
        # 'observe_classes': [0, '0_backdoor', '0_clean', '0_sample', 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'observe_classes': [0, '0_backdoor', '0_clean', '0_sample', 1, 2, 3],
        'mi_estimate_epochs': 450,
        'mi_batch_size': {
            'inputs-vs-outputs': 400,
            'outputs-vs-Y': 1024
        }
    }
    
    def __init__(self, **overrides):
        """Initialize configuration with optional overrides
        
        Args:
            **overrides: Keyword arguments to override default settings
        """
        self.config = {**self.DEFAULT, **overrides}
        
    def __getitem__(self, key):
        """Allow dictionary-style access to config values"""
        return self.config[key]
    
    def get(self, key, default=None):
        """Get config value with optional default"""
        return self.config.get(key, default)
    
    @property
    def device(self):
        """Get the device to use for training"""
        if self.config['device'] == 'cuda' and not torch.cuda.is_available():
            return 'cpu'
        return self.config['device']

    @classmethod
    def from_args(cls, args):
        # 支持 Namespace 或 dict
        if isinstance(args, dict):
            overrides = args
        else:
            overrides = vars(args)
        return cls(**{k: v for k, v in overrides.items() if v is not None})

    def __getattr__(self, key):
        # 支持 config.xxx 访问
        if key == 'config':
            return self.__dict__['config']
        if key in self.config:
            return self.config[key]
        raise AttributeError(f"'TrainingConfig' object has no attribute '{key}'")

# train one epoch
def train_loop(dataloader, model, loss_fn, optimizer, num_classes):
    size, num_batches = dataloader.batch_size, len(dataloader)
    model.train()
    epoch_acc = 0.0
    class_losses = torch.zeros(num_classes).to(next(model.parameters()).device)
    class_counts = torch.zeros(num_classes).to(next(model.parameters()).device)

    # 收集数据
    # 预分配张量存储数据
    total_samples = 50000
    t = torch.zeros((total_samples, 128), device=device)  # 特征维度为512
    pred_all = torch.zeros((total_samples, num_classes), device=device)
    labels_all = torch.zeros((total_samples), dtype=torch.long, device=device)
    is_backdoor_all = torch.zeros((total_samples), dtype=torch.long, device=device)
    current_idx = 0

    # 注册钩子函数到最后一个 BasicBlock
    # hook_handle = model.layer4[-1].register_forward_hook(hook)
    # hook_handle = model.layer5[-1].register_forward_hook(hook)
    # hook_handle = model.to_latent.register_forward_hook(hook)

    for batch, (X, Y, is_backdoor) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        epoch_acc += get_acc(pred, Y)

        # 计算每个类别的损失
        for c in range(num_classes):
            mask = (Y == c)
            if mask.sum() > 0:
                class_losses[c] += loss_fn(pred[mask], Y[mask]).item() * mask.sum().item()
                class_counts[c] += mask.sum().item()
        
        with torch.no_grad():
            # M_output = F.adaptive_avg_pool2d(last_conv_output, 1)
            M_output = model.cls_embedding
            M_output = M_output.view(M_output.shape[0], -1)
        
        batch_size = len(Y)
        end_idx = current_idx + batch_size

        t[current_idx:end_idx] = M_output
        pred_all[current_idx:end_idx] = pred
        labels_all[current_idx:end_idx] = Y
        is_backdoor_all[current_idx:end_idx] = is_backdoor
        current_idx = end_idx
    
    # 在计算MI之前移除钩子
    # hook_handle.remove()
    
    # 裁剪张量到实际大小
    t = t[:current_idx].detach()
    pred_all = pred_all[:current_idx].detach()
    labels_all = labels_all[:current_idx]
    is_backdoor_all = is_backdoor_all[:current_idx]

    avg_acc = 100 * (epoch_acc / num_batches)
    
    # 计算每个类别的平均损失
    class_losses = class_losses / class_counts
    class_losses = class_losses.cpu().numpy()

    print(f'Train acc: {avg_acc:.2f}%')
    for c in range(num_classes):
        print(f'Class {c} loss: {class_losses[c]:.4f}')

    return avg_acc, class_losses, t, pred_all, labels_all, is_backdoor_all
    # return avg_acc, class_losses

def test_loop(dataloader, model, loss_fn):
    # Set the models to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = dataloader.batch_size
    num_batches = len(dataloader)
    total = size*num_batches
    test_loss, correct = 0, 0

    # Evaluating the models with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= total
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, (100 * correct)

# 定义钩子函数
def hook(module, input, output):
    global last_conv_output
    last_conv_output = output.detach()

def estimate_mi(args, device, flag, model_state_dict, sample_loader, class_idx, EPOCHS=50, mode='infoNCE'):
    if args['model'] == 'resnet18':
        model = ResNet18(num_classes=10, noise_std_xt=args['noise_std_xt'], noise_std_ty=args['noise_std_ty'])
        model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = torch.nn.Linear(512, 10) # 将最后的全连接层改掉
        model.load_state_dict(model_state_dict)
    elif args['model'] == 'vgg16':
        model = VGG16(num_classes=10, noise_std_xt=args['noise_std_xt'], noise_std_ty=args['noise_std_ty'])
        model.load_state_dict(model_state_dict)
    elif args['model'] == 'vit':
        model = ViT(
            image_size=32,         # CIFAR-10 image size
            patch_size=4,          # 4x4 patches
            num_classes=10,
            dim=128,
            depth=6,
            heads=8,
            mlp_dim=256,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.1,
            noise_std_xt=args['noise_std_xt'],
            noise_std_ty=args['noise_std_ty']
        )
        model.load_state_dict(model_state_dict)
    else:
        raise ValueError(f"Unsupported model: {args['model']}")
    model.to(device).eval()

    # LR = 1e-5
    initial_lr = 1e-4
    if flag == 'inputs-vs-outputs':
        Y_dim, Z_dim = 128, 3072  # M的维度, X的维度
    elif flag == 'outputs-vs-Y':
        initial_lr = 2e-4
        Y_dim, Z_dim = 10, 128  # Y的维度, M的维度
    else:
        raise ValueError('Not supported!')
    
    T = TNet(in_dim=Y_dim + Z_dim, hidden_dim=128).to(device)
    # T = torch.nn.DataParallel(T)  # 使用 DataParallel
    optimizer = torch.optim.AdamW(T.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    M = []
    
    # 使用tqdm.tqdm而不是tqdm.auto，并设置position参数
    position = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    progress_bar = tqdm(
        range(EPOCHS),
        desc=f"class {class_idx}",
        position=position,
        leave=True,
        ncols=100
    )
    
    # 注册钩子函数到最后一个 BasicBlock
    # global last_conv_output
    # last_conv_output = None
    # hook_handle = model.layer4[-1].register_forward_hook(hook)
    # hook_handle = model.layer5[-1].register_forward_hook(hook)
    # hook_handle = model.to_latent.register_forward_hook(hook)

    for epoch in progress_bar:
        epoch_losses = []
        for batch, (X, _Y) in enumerate(sample_loader):
            if batch > len(sample_loader)//2 and class_idx == 0:
                continue
            X, _Y = X.to(device), _Y.to(device)
            with torch.no_grad():
                with autocast(device_type="cuda"):
                    Y_predicted = model(X)
                # if last_conv_output is None:
                #     raise ValueError("last_conv_output is None. Ensure the hook is correctly registered and the model is correctly defined.")
                # 对 last_conv_output 进行全局平均池化
                # M_output = F.adaptive_avg_pool2d(last_conv_output, 1)
                M_output = model.cls_embedding
                # M_output = M_output.view(M_output.shape[0], -1)
            if flag == 'inputs-vs-outputs':
                X_flat = torch.flatten(X, start_dim=1)
                # print(f'X_flat.shape: {X_flat.shape}, M_output.shape: {M_output.shape}')
                with autocast(device_type="cuda"):
                    loss, _ = compute_infoNCE(T, M_output, X_flat, num_negative_samples=128)
            elif flag == 'outputs-vs-Y':
                Y = Y_predicted
                with autocast(device_type="cuda"):
                    loss, _ = compute_infoNCE(T, Y, M_output, num_negative_samples=128)

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                print(f"Skipping batch due to invalid loss: {loss.item()}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(T.parameters(), 5)
            optimizer.step()
            epoch_losses.append(loss.item())
        
        if not epoch_losses:
            M.append(float('nan'))
            continue
        
        avg_loss = np.mean(epoch_losses)
        M.append(-avg_loss)
     
        # 更新进度条
        progress_bar.set_postfix({'mi_estimate': -avg_loss})
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        # 提前停止检查
        if dynamic_early_stop(M, delta=1e-2):
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # 清理进度条
    progress_bar.close()
    # 清理缓存
    torch.cuda.empty_cache()
    gc.collect()
    return M


def estimate_mi_wrapper(args):
    base_args, flag, model_state_dict, class_idx, EPOCHS, mode = args
    # base_args 现在是 dict，不是 TrainingConfig 实例
    # 如果需要 TrainingConfig 实例，可以这样恢复：
    # base_args = TrainingConfig.from_args(base_args)
    # 但推荐直接用 dict
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # if isinstance(class_idx, int) and int(class_idx) > 5:
    #     device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToDevice(device)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline,
    }
    sample_loader_path = f"{base_args['sample_data_path']}/class_{class_idx}.beton"
    
    sample_batch_size = 400 if flag == "inputs-vs-outputs" else 1024
    sample_loader = Loader(sample_loader_path, batch_size=sample_batch_size, num_workers=20,
                            order=OrderOption.RANDOM, pipelines=pipelines, drop_last=False)
    
    return estimate_mi(base_args, device, flag, model_state_dict, sample_loader, class_idx, EPOCHS, mode)

def train(args, flag='inputs-vs-outputs', mode='infoNCE'):
    """ flag = inputs-vs-outputs or outputs-vs-Y """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256  
    learning_rate = 0.1

    # 动态设置 num_workers
    num_workers = 20

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

    train_dataloader_path = config['train_data_path']
    train_dataloader = Loader(train_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                              order=OrderOption.RANDOM, os_cache=True, pipelines=pipelines, drop_last=False, seed=0)

    test_dataloader_path = config['test_data_path']
    test_dataloader = Loader(test_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                             order=OrderOption.RANDOM, pipelines=test_pipelines, seed=0)
    
    test_poison_data = np.load(config['test_poison_data_path'])
    test_poison_dataset = TensorDataset(
        torch.tensor(test_poison_data['arr_0'], dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(test_poison_data['arr_1'], dtype=torch.long)
    )
    test_poison_dataloader = DataLoader(test_poison_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    

    num_classes = config['num_classes']
    if config['model'] == 'resnet18':
        model = ResNet18(num_classes=num_classes, noise_std_xt=config['noise_std_xt'], noise_std_ty=config['noise_std_ty'])  
        model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = torch.nn.Linear(512, num_classes) # 将最后的全连接层改掉
    elif config['model'] == 'vgg16':
        model = VGG16(num_classes=num_classes, noise_std_xt=config['noise_std_xt'], noise_std_ty=config['noise_std_ty'])
    elif config['model'] == 'vit':
        model = ViT(
            image_size=32,         # CIFAR-10 image size
            patch_size=4,          # 4x4 patches
            num_classes=num_classes,
            dim=128,               # embedding dimension
            depth=6,               # number of transformer layers
            heads=8,               # number of attention heads
            mlp_dim=256,           # MLP hidden dimension
            pool='cls',            # use CLS token
            channels=3,            # RGB
            dim_head=64,           # dimension per head
            dropout=0.1,           # dropout rate
            emb_dropout=0.1,       # embedding dropout
            noise_std_xt=config['noise_std_xt'],
            noise_std_ty=config['noise_std_ty']
        )
    else:
        raise ValueError(f"Unsupported model: {config['model']}")
    # model = nn.DataParallel(model)  # 使用 DataParallel
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # 使用 StepLR 调整学习率，每10个epoch，lr乘0.5
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_accuracy = 0
    best_model = None
    epochs = config['epochs']
    MI_inputs_vs_outputs = {class_idx: [] for class_idx in config['observe_classes']}
    MI_Y_vs_outputs = {class_idx: [] for class_idx in config['observe_classes']}
    class_losses_list = []
    previous_test_loss = float('inf')
    
    # 初始化 swanlab
    swanlab.init(
        project=f"MI-Analysis-sampleLoader-{config['attack_type']}",
        name=f"{config['model']}_xt{config['noise_std_xt']}_ty{config['noise_std_ty']}_{config['outputs_dir'].split('/')[-2]}_{config['train_data_path'].split('/')[-2]}",
        config={
            "model": config['model'],
            "noise_std_xt": config['noise_std_xt'],
            "noise_std_ty": config['noise_std_ty'],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "num_workers": num_workers,
            "observe_classes": config['observe_classes'],
            "train_data_path": config['train_data_path'],
            "test_data_path": config['test_data_path']
        }
    )

    for epoch in range(1, epochs + 1):
        print(f"------------------------------- Epoch {epoch} -------------------------------")
        train_acc, class_losses, t, preds, labels, is_backdoor = train_loop(train_dataloader, model, loss_fn, optimizer, num_classes)
        # train_acc, class_losses = train_loop(train_dataloader, model, loss_fn, optimizer, num_classes)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        _asr = calculate_asr(model, test_poison_dataloader, 0, device)       
        class_losses_list.append(class_losses)

        # Visualize t using t-SNE
        # if epoch in [5, 10, 20, 40, 60, 80, 120]:
        #     plot_tsne(t, labels, is_backdoor, epoch, config['outputs_dir'])
            # plot_tsne(preds, labels, is_backdoor, epoch, args.outputs_dir, prefix='preds')
            
        # 创建一个包含所有类别损失的图表
        swanlab.log({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "attack_success_rate": _asr,
        }, step=epoch)

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = copy.deepcopy(model)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        # 调整学习率
        scheduler.step(test_loss)
        
        # 检查是否应该计算互信息
        # should_compute_mi = ((t % pow(2, t//10) == 0) or t%10==0) and test_loss < previous_test_loss
        # should_compute_mi = (t % pow(2, t//10) == 0) and (test_loss < previous_test_loss if t < 10 else True)
        # should_compute_mi = test_loss < previous_test_loss
        # should_compute_mi = t==1 or t==8 or t==15 or t==25 or t==40 or t==60
        # should_compute_mi = epoch in [1, 5, 10, 20, 40, 60, 80, 100, 120]
        # should_compute_mi = epoch in [1, 3, 8, 10, 20, 30, 50]
        # should_compute_mi = epoch in [1, 20, 40]
        # should_compute_mi = t==20 or t==80
        should_compute_mi = False
        if should_compute_mi:
            print(f"------------------------------- Epoch {epoch} -------------------------------")
            mi_inputs_vs_outputs_dict = {}
            mi_Y_vs_outputs_dict = {}
            model_state_dict = model.state_dict()
            # 创建一个进程池
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(config['observe_classes'])) as executor:
                # 计算 I(X,T) 和 I(T,Y)
                compute_args = [
                    (config.config, 'inputs-vs-outputs', model_state_dict, class_idx, 300, mode)
                    for class_idx in config['observe_classes']
                ]
                results_inputs_vs_outputs = list(executor.map(estimate_mi_wrapper, compute_args))

            with concurrent.futures.ProcessPoolExecutor(max_workers=len(config['observe_classes'])) as executor:    
                compute_args = [(config.config, 'outputs-vs-Y', model_state_dict, class_idx, 300, mode) 
                                for class_idx in config['observe_classes']]
                results_Y_vs_outputs = list(executor.map(estimate_mi_wrapper, compute_args))

            # 处理结果
            for class_idx, result in zip(config['observe_classes'], results_inputs_vs_outputs):
                mi_inputs_vs_outputs = result
                mi_inputs_vs_outputs_dict[class_idx] = mi_inputs_vs_outputs
                MI_inputs_vs_outputs[class_idx].append(mi_inputs_vs_outputs)

            for class_idx, result in zip(config['observe_classes'], results_Y_vs_outputs):
                mi_Y_vs_outputs = result
                mi_Y_vs_outputs_dict[class_idx] = mi_Y_vs_outputs
                MI_Y_vs_outputs[class_idx].append(mi_Y_vs_outputs)

            # 保存 MI 图到 swanlab
            plot_and_save_mi(mi_inputs_vs_outputs_dict, 'inputs-vs-outputs', config['outputs_dir'], epoch)
            plot_and_save_mi(mi_Y_vs_outputs_dict, 'outputs-vs-Y', config['outputs_dir'], epoch)

            np.save(f"{config['outputs_dir']}/infoNCE_MI_I(X,T).npy", MI_inputs_vs_outputs)
            np.save(f"{config['outputs_dir']}/infoNCE_MI_I(Y,T).npy", MI_Y_vs_outputs)

            # 上传图片到 swanlab
            swanlab.log({
                f"I(X;T)_estimation": swanlab.Image(os.path.join(config['outputs_dir'], f'mi_plot_inputs-vs-outputs_epoch_{epoch}.png')),
                f"I(T;Y)_estimation": swanlab.Image(os.path.join(config['outputs_dir'], f'mi_plot_outputs-vs-Y_epoch_{epoch}.png'))
            }, step=epoch)

        # 更新前一个epoch的test_loss
        previous_test_loss = test_loss

    plot_train_loss_by_class(class_losses_list, epoch, num_classes, config['outputs_dir'])
    swanlab.log({
        "train_loss_by_class": swanlab.Image(os.path.join(config['outputs_dir'], 'train_loss_by_class_plot.png'))
    })

    swanlab.finish()
    return MI_inputs_vs_outputs, MI_Y_vs_outputs, best_model


def ob_infoNCE(config):
    outputs_dir = config['outputs_dir']
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    infoNCE_MI_log_inputs_vs_outputs, infoNCE_MI_log_Y_vs_outputs, best_model = train(config, 'inputs-vs-outputs', 'infoNCE')
     
    # 保存最佳模型
    # torch.save(best_model, os.path.join(args.outputs_dir, 'best_model.pth'))

    # 检查并保存 infoNCE_MI_log_inputs_vs_outputs
    infoNCE_MI_log_inputs_vs_outputs = np.array(infoNCE_MI_log_inputs_vs_outputs, dtype=object)
    np.save(f'{outputs_dir}/infoNCE_MI_I(X,T).npy', infoNCE_MI_log_inputs_vs_outputs)
    print(f'saved in {outputs_dir}/infoNCE_MI_I(X,T).npy')
    
    # 检查并保存 infoNCE_MI_log_Y_vs_outputs
    infoNCE_MI_log_Y_vs_outputs = np.array(infoNCE_MI_log_Y_vs_outputs, dtype=object)
    np.save(f'{outputs_dir}/infoNCE_MI_I(Y,T).npy', infoNCE_MI_log_Y_vs_outputs)
    print(f'saved in {outputs_dir}/infoNCE_MI_I(Y,T).npy')

if __name__ == '__main__':
    device = torch.device('cuda')
    mp.set_start_method('spawn', force=True)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, default='results/ob_infoNCE_06_22', help='output_dir')
    parser.add_argument('--train_data_path', type=str, default='0', help='path of training data')
    parser.add_argument('--test_data_path', type=str, default='0', help='path of test data')
    parser.add_argument('--test_poison_data_path', type=str, default="data/cifar10/badnet/0.1/poisoned_test_data.npz", help='path of poisoned test data')
    parser.add_argument('--sample_data_path', type=str, default='', help='path of sample dataloader')
    parser.add_argument('--model', type=str, default='vit', help='model')

    # parser.add_argument('--observe_classes', type=list, default=[0,1,2,3,4,5,6,7,8,9], help='class')
    parser.add_argument('--observe_classes', type=list, default=[0,'0_backdoor','0_clean','0_sample',1,2,3], help='class')
    args = parser.parse_args()
    config = TrainingConfig.from_args(args)
    ob_infoNCE(config)
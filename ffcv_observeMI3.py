import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.resnet import ResNet18
from model.TNet import TNet
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

proc_name = 'lover'
setproctitle.setproctitle(proc_name)


def get_acc(outputs, labels):
    """calculate acc"""
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0] * 1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num
    return acc


# train one epoch
def train_loop(dataloader, model, loss_fn, optimizer, num_classes):
    size, num_batches = dataloader.batch_size, len(dataloader)
    model.train()
    epoch_acc = 0.0
    class_losses = torch.zeros(num_classes).to(next(model.parameters()).device)
    class_counts = torch.zeros(num_classes).to(next(model.parameters()).device)

    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        epoch_acc += get_acc(pred, y)

        # 计算每个类别的损失
        for c in range(num_classes):
            mask = (y == c)
            if mask.sum() > 0:
                class_losses[c] += loss_fn(pred[mask], y[mask]).item() * mask.sum().item()
                class_counts[c] += mask.sum().item()

    avg_acc = 100 * (epoch_acc / num_batches)
    
    # 计算每个类别的平均损失
    class_losses = class_losses / class_counts
    class_losses = class_losses.cpu().numpy()

    print(f'Train acc: {avg_acc:.2f}%')
    for c in range(num_classes):
        print(f'Class {c} loss: {class_losses[c]:.4f}')

    return avg_acc, class_losses

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


def compute_infoNCE(T, Y, Z, num_negative_samples=512):
    batch_size = Y.shape[0]
    # 随机选择负样本
    negative_indices = torch.randint(0, batch_size, (batch_size, num_negative_samples), device=Y.device)
    Z_negative = Z[negative_indices]
    
    # 计算正样本的得分
    t_positive = T(Y, Z).squeeze() # (batch_size, )
    # 计算负样本的得分
    Y_expanded = Y.unsqueeze(1).expand(-1, num_negative_samples, -1) # (batch_size, num_negative_samples, Y.dim)
    t_negative = T(Y_expanded.reshape(-1, Y.shape[1]), Z_negative.reshape(-1, Z.shape[1])) # (batch_size*num_negative_samples, )
    t_negative = t_negative.view(batch_size, num_negative_samples) # (batch_size, num_negative_samples)
    
    # 计算 InfoNCE loss
    logits = torch.cat([t_positive.unsqueeze(1), t_negative], dim=1).to(Y.device)  # (batch_size, num_negative_samples+1)
    # 创建标签，正样本在第0位
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=Y.device)  # (batch_size,)

    # 使用交叉熵损失来计算 InfoNCE 损失
    loss = -math.log(num_negative_samples+1) + F.cross_entropy(logits, labels)
    
    # 计算每个样本的得分差值（正样本得分减去平均负样本得分）
    sample_score_diffs = t_positive - t_negative.mean(dim=1)
    
    return logits, loss, sample_score_diffs.detach()

def dynamic_early_stop(M, patience=50, delta=1e-3):
    if len(M) > patience:
        recent_M = M[-patience:]
        if max(recent_M) - min(recent_M) < delta:
            return True
    return False

# 定义钩子函数
def hook(module, input, output):
    global last_conv_output
    last_conv_output = output

def estimate_mi(model, flag, sample_loader, EPOCHS=50, mode='infoNCE'):
    # LR = 1e-5
    initial_lr = 1e-4
    model.eval()
    if flag == 'inputs-vs-outputs':
        Y_dim, Z_dim = 512, 3072  # M的维度, X的维度
    elif flag == 'outputs-vs-Y':
        Y_dim, Z_dim = 10, 512  # Y的维度, M的维度
    else:
        raise ValueError('Not supported!')
    
    T = TNet(in_dim=Y_dim + Z_dim, hidden_dim=256).to(device)
    # T = torch.nn.DataParallel(T)  # 使用 DataParallel
    optimizer = torch.optim.AdamW(T.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    M = []

    # 注册钩子函数到最后一个 BasicBlock
    global last_conv_output
    last_conv_output = None
    hook_handle = model.layer4[-1].register_forward_hook(hook)

    sample_score_diffs = torch.zeros(len(sample_loader)*sample_loader.batch_size).to(device)

    for epoch in range(EPOCHS):
        print(f"------------------------------- MI-Esti-Epoch {epoch + 1}-{mode} -------------------------------")
        L = []
        for batch, (X, _Y) in enumerate(sample_loader):
            X, _Y = X.to(device), _Y.to(device)
            with torch.no_grad():
                with autocast(device_type="cuda"):
                    # Y = F.one_hot(_Y, num_classes=10)
                    Y_predicted = model(X)
                    if last_conv_output is None:
                        raise ValueError("last_conv_output is None. Ensure the hook is correctly registered and the model is correctly defined.")
                    # 对 last_conv_output 进行全局平均池化
                    M_output = F.adaptive_avg_pool2d(last_conv_output, 1)
                    M_output = M_output.view(M_output.shape[0], -1)
            if flag == 'inputs-vs-outputs':
                X_flat = torch.flatten(X, start_dim=1)
                # print(f'X_flat.shape: {X_flat.shape}, M_output.shape: {M_output.shape}')
                _, loss, batch_scores = compute_infoNCE(T, M_output, X_flat)
            elif flag == 'outputs-vs-Y':
                # Y = Y_predicted
                Y = _Y
                _, loss, batch_scores = compute_infoNCE(T, Y, M_output)
            
            # 更新样本得分
            if epoch == EPOCHS - 1:
                start_idx = batch * sample_loader.batch_size
                end_idx = start_idx + X.shape[0]
                sample_score_diffs[start_idx:end_idx] += batch_scores

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                print(f"Skipping batch due to invalid loss: {loss.item()}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(T.parameters(), 5)
            optimizer.step()
            L.append(loss.item())
        
        if not L:
            M.append(float('nan'))
            continue
        
        avg_loss = np.mean(L)
        print(f'[{mode}] loss:', avg_loss, max(L), min(L))
        M.append(-avg_loss)  # 负的 InfoNCE loss 作为互信息的下界估计
        print(f'[{mode}] mi estimate:', -avg_loss)
        
        # Update the learning rate
        scheduler.step(avg_loss)
        
        if dynamic_early_stop(M, delta=1e-2 if flag == 'inputs-vs-outputs' else 1e-3):
            print(f'Early stopping at epoch {epoch + 1}')
            break
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()

    return M, sample_score_diffs


def plot_and_save_mi(mi_values_dict, mode, output_dir, epoch):
    plt.figure(figsize=(10, 6))
    for class_idx, mi_values in mi_values_dict.items():
        epochs = range(1, len(mi_values) + 1)
        plt.plot(epochs, mi_values, label=f'Class {class_idx}')
    
    plt.xlabel('Epochs')
    plt.ylabel('MI Value')
    plt.title(f'MI Estimation over Epochs ({mode}) - Training Epoch {epoch}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'mi_plot_{mode}_epoch_{epoch}.png'))
    plt.close()

def plot_train_acc(train_accuracies, test_accuracies, epochs, outputs_dir):
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Training')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(outputs_dir + '/accuracy_plot.png')

def plot_train_loss_by_class(train_losses, epochs, num_classes, outputs_dir):
    plt.figure(figsize=(12, 8))
    for c in range(num_classes):
        plt.plot(range(1, epochs + 1), [losses[c] for losses in train_losses], label=f'Class {c}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss by Class over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outputs_dir, 'train_loss_by_class_plot.png'))
    plt.close()

def compute_class_accuracy(model, dataloader, num_classes):
    model.eval()
    correct = [0] * num_classes
    total = [0] * num_classes
    
    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(y)):
                label = y[i].item()
                total[label] += 1
                if predicted[i] == label:
                    correct[label] += 1
    
    accuracies = [100 * correct[i] / total[i] if total[i] > 0 else 0 for i in range(num_classes)]
    return accuracies

def update_class_weights(accuracies, current_weights, alpha=0.5, epsilon=1e-8):
    inv_accuracies = 1.0 / (np.array(accuracies) + epsilon)  # 添加一个小的epsilon值来避免除以零
    new_weights = inv_accuracies / np.sum(inv_accuracies) * len(accuracies)
    updated_weights = alpha * new_weights + (1 - alpha) * current_weights
    return torch.tensor(updated_weights, dtype=torch.float32)

def train(args, flag='inputs-vs-outputs', mode='infoNCE'):
    """ flag = inputs-vs-outputs or outputs-vs-Y """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512  
    learning_rate = 0.1

    # 动态设置 num_workers
    num_workers = 16

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToDevice(device)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    label_pipeline_sample = [ToTensor(), ToDevice(device)]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }
    pipelines_sample = {
        'image': image_pipeline,
        'label': label_pipeline_sample
    }

    sample_dataloaders = {}
    for class_idx in args.observe_classes:
        sample_dataloader_path = f"{args.sample_data_path}_class_{class_idx}.beton"
        sample_dataloaders[class_idx] = Loader(sample_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                                               order=OrderOption.RANDOM, pipelines=pipelines_sample)
    train_dataloader_path = args.train_data_path
    train_dataloader = Loader(train_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                              order=OrderOption.RANDOM, pipelines=pipelines, drop_last=False)

    test_dataloader_path = args.test_data_path
    test_dataloader = Loader(test_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                             order=OrderOption.RANDOM, pipelines=pipelines)

    num_classes = 10
    model = ResNet18(num_classes=num_classes)  
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, num_classes) # 将最后的全连接层改掉
    # model = nn.DataParallel(model)  # 使用 DataParallel
    model.to(device)
    model.train()

    initial_weights = torch.tensor([1.0] * num_classes).to(device)
    class_weights = initial_weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # 使用 StepLR 调整学习率，每10个epoch，lr乘0.5
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_accuracy = 0
    best_model = None
    epochs = 80
    MI_inputs_vs_outputs = {class_idx: [] for class_idx in args.observe_classes}
    MI_Y_vs_outputs = {class_idx: [] for class_idx in args.observe_classes}
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    previous_test_loss = float('inf')
    sample_scores_dict = {class_idx: None for class_idx in args.observe_classes}

    for t in range(1, epochs + 1):
        print(f"------------------------------- Epoch {t} -------------------------------")
        train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_acc)

        # # 保存最佳模型
        # if test_acc > best_accuracy:
        #     best_accuracy = test_acc
        #     best_model = copy.deepcopy(model)
        #     print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        # 调整学习率
        scheduler.step(test_loss)
        
        # 检查是否应该计算互信息
        # should_compute_mi = (t % pow(2, t//10) == 0) and test_loss < previous_test_loss
        should_compute_mi = (t % pow(2, t//10) == 0) and test_loss < previous_test_loss and 3<t<10
        # should_compute_mi = test_loss < previous_test_loss
        if should_compute_mi:
            print(f"------------------------------- Epoch {t} -------------------------------")
            mi_inputs_vs_outputs_dict = {}
            mi_Y_vs_outputs_dict = {}
            for class_idx, sample_dataloader in sample_dataloaders.items():
                # mi_inputs_vs_outputs, sample_scores = estimate_mi(model, 'inputs-vs-outputs', sample_dataloader, EPOCHS=350, mode=mode)
                # MI_inputs_vs_outputs[class_idx].append(mi_inputs_vs_outputs)
                # mi_inputs_vs_outputs_dict[class_idx] = mi_inputs_vs_outputs
                
                mi_Y_vs_outputs, sample_scores = estimate_mi(model, 'outputs-vs-Y', sample_dataloader, EPOCHS=250, mode=mode)
                MI_Y_vs_outputs[class_idx].append(mi_Y_vs_outputs)
                mi_Y_vs_outputs_dict[class_idx] = mi_Y_vs_outputs
                sample_scores_dict[class_idx] = sample_scores

            # plot_and_save_mi(mi_inputs_vs_outputs_dict, 'inputs-vs-outputs', args.outputs_dir, t)
            plot_and_save_mi(mi_Y_vs_outputs_dict, 'outputs-vs-Y', args.outputs_dir, t)
            
            # 分析样本得分
            analyze_sample_scores(sample_scores_dict, args.outputs_dir, t)
        
        # 更新前一个epoch的test_loss
        previous_test_loss = test_loss

    np.save(os.path.join(args.outputs_dir, 'train_losses_by_class.npy'), np.array(train_losses))
 
    plot_train_acc(train_accuracies, test_accuracies, epochs, args.outputs_dir)
    plot_train_loss_by_class(train_losses, epochs, num_classes, args.outputs_dir)

    # 训练完成后
    print("Computing class-wise accuracies...")
    if not best_model:
        best_model = model
    train_accuracies = compute_class_accuracy(best_model, train_dataloader, num_classes)
    test_accuracies = compute_class_accuracy(best_model, test_dataloader, num_classes)

    print("Train accuracies per class:")
    for i, acc in enumerate(train_accuracies):
        print(f"Class {i}: {acc:.2f}%")

    print("\nTest accuracies per class:")
    for i, acc in enumerate(test_accuracies):
        print(f"Class {i}: {acc:.2f}%")

    return MI_inputs_vs_outputs, MI_Y_vs_outputs, best_model, train_accuracies, test_accuracies

def analyze_sample_scores(sample_scores_dict, output_dir, epoch):
    # 将所有类别的得分合并到一个列表中
    all_scores = []
    all_class_indices = []
    all_sample_indices = []
    
    for class_idx, scores in sample_scores_dict.items():
        all_scores.append(scores.cpu())  # 确保scores在CPU上
        all_class_indices.extend([class_idx] * len(scores))
        all_sample_indices.append(torch.arange(len(scores)))
    
    all_scores = torch.cat(all_scores)
    all_class_indices = torch.tensor(all_class_indices)
    all_sample_indices = torch.cat(all_sample_indices)
    
    # 计算所有得分的统计信息
    mean_score = all_scores.mean().item()
    std_score = all_scores.std().item()
    
    # 找出得分异常高的样本（例如，高于平均值两个标准差）
    threshold = mean_score + 2 * std_score
    suspicious_mask = all_scores > threshold
    suspicious_scores = all_scores[suspicious_mask]
    suspicious_classes = all_class_indices[suspicious_mask]
    suspicious_indices = all_sample_indices[suspicious_mask]
    
    print(f"Overall - Mean score: {mean_score:.4f}, Std: {std_score:.4f}")
    print(f"Number of suspicious samples: {len(suspicious_scores)}")
    
    # 保存可疑样本的信息
    suspicious_info = np.column_stack((suspicious_classes.cpu().numpy(), 
                                       suspicious_indices.cpu().numpy(), 
                                       suspicious_scores.cpu().numpy()))
    np.save(os.path.join(output_dir, f'suspicious_samples_epoch_{epoch}.npy'), suspicious_info)
    
    # 绘制每个类别的得分箱线图
    plt.figure(figsize=(12, 6))
    plt.boxplot([scores.cpu().numpy() for scores in sample_scores_dict.values()], labels=sample_scores_dict.keys())
    plt.title(f'Score Distribution by Class - Epoch {epoch}')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.savefig(os.path.join(output_dir, f'score_distribution_by_class_epoch_{epoch}.png'))
    plt.close()
    
    # 输出每个类别中可疑样本的数量
    for class_idx in sample_scores_dict.keys():
        class_suspicious_count = (suspicious_classes == class_idx).sum().item()
        print(f"Class {class_idx} - Number of suspicious samples: {class_suspicious_count}")

    return suspicious_info

def ob_infoNCE(args):
    outputs_dir = args.outputs_dir
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    infoNCE_MI_log_inputs_vs_outputs, infoNCE_MI_log_Y_vs_outputs, best_model, train_accuracies, test_accuracies = train(args, 'inputs-vs-outputs', 'infoNCE')
     
    # 保存最佳模型
    # torch.save(best_model, os.path.join(args.outputs_dir, 'best_model.pth'))

    # 保存准确率数据
    np.save(os.path.join(outputs_dir, 'train_class_accuracies.npy'), train_accuracies)
    np.save(os.path.join(outputs_dir, 'test_class_accuracies.npy'), test_accuracies)

    # 检查并保存 infoNCE_MI_log_inputs_vs_outputs
    for class_idx, mi_log in infoNCE_MI_log_inputs_vs_outputs.items():
        mi_log = np.array(mi_log, dtype=object)  # 确保 mi_log 是 numpy 数组
        print(f'Saving infoNCE_MI_I(X,T)_class_{class_idx}.npy with shape {mi_log.shape}')
        np.save(f'{outputs_dir}/infoNCE_MI_I(X,T)_class_{class_idx}.npy', mi_log)
        print(f'{outputs_dir}/infoNCE_MI_I(X,T)_class_{class_idx}.npy 已保存')
    
    # 检查并保存 infoNCE_MI_log_Y_vs_outputs
    for class_idx, mi_log in infoNCE_MI_log_Y_vs_outputs.items():
        mi_log = np.array(mi_log, dtype=object)  # 确保 mi_log 是 numpy 数组
        print(f'Saving infoNCE_MI_I(Y,T)_class_{class_idx}.npy with shape {mi_log.shape}')
        np.save(f'{outputs_dir}/infoNCE_MI_I(Y,T)_class_{class_idx}.npy', mi_log)
        print(f'{outputs_dir}/infoNCE_MI_I(Y,T)_class_{class_idx}.npy 已保存')
if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, default='results/ob_infoNCE_06_22', help='output_dir')
    parser.add_argument('--sampling_datasize', type=str, default='1000', help='sampling_datasize')
    parser.add_argument('--training_epochs', type=str, default='100', help='training_epochs')
    parser.add_argument('--batch_size', type=str, default='256', help='batch_size')
    parser.add_argument('--learning_rate', type=str, default='1e-5', help='learning_rate')
    parser.add_argument('--mi_estimate_epochs', type=str, default='300', help='mi_estimate_epochs')
    parser.add_argument('--mi_estimate_lr', type=str, default='1e-6', help='mi_estimate_lr')
    parser.add_argument('--class', type=str, default='0', help='class')
    parser.add_argument('--train_data_path', type=str, default='0', help='class')
    parser.add_argument('--test_data_path', type=str, default='0', help='class')
    parser.add_argument('--sample_data_path', type=str, default='data/badnet/train_data', help='class')
    parser.add_argument('--observe_classes', type=list, default=[0,1,2,3,4,5,6,7,8,9], help='class')
    args = parser.parse_args()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ob_DV()
    ob_infoNCE(args)
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
def train_loop(dataloader, model, loss_fn, optimizer):
    size, num_batches = dataloader.batch_size, len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    epoch_acc, epoch_loss = 0.0, 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        epoch_acc += get_acc(pred, y)
        epoch_loss += loss.data
    print('Train loss: %.4f, Train acc: %.2f' % (epoch_loss/size, 100 * (epoch_acc / num_batches)))
    return 100 * (epoch_acc / num_batches)

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


# def compute_infoNCE(T, Y, Z, t):
#     Y_ = Y.repeat_interleave(Y.shape[0], dim=0)
#     Z_ = Z.tile(Z.shape[0], 1)
#     t2 = T(Y_, Z_).view(Y.shape[0], Y.shape[0], -1)
#     t2 = t2.exp().mean(dim=1).log()  # mean over j
#     # assert t.shape == t2.shape
#     loss = -(t.mean() - t2.mean())
#     return t2, loss
# def stable_log_sum_exp(logits, dim=1):
#     max_logits, _ = torch.max(logits, dim=dim, keepdim=True)
#     stable_logits = logits - max_logits
#     log_sum_exp = (stable_logits.exp().mean(dim=dim)).log() + max_logits.squeeze(dim)
#     return log_sum_exp
# def compute_infoNCE(T, Y, Z, t, num_negative_samples=256):
#     batch_size = Y.shape[0]
    
#     # 随机选择负样本
#     negative_indices = torch.randint(0, batch_size, (batch_size, num_negative_samples), device=Y.device)
#     Z_negative = Z[negative_indices]
    
#     # 计算正样本的得分
#     t_positive = t.squeeze()
#     # 计算负样本的得分
#     Y_expanded = Y.unsqueeze(1).expand(-1, num_negative_samples, -1)
#     t_negative = T(Y_expanded.reshape(-1, Y.shape[1]), Z_negative.reshape(-1, Z.shape[1]))
#     t_negative = t_negative.view(batch_size, num_negative_samples)
    
#     # 计算 InfoNCE loss
#     logits = torch.cat([t_positive.unsqueeze(1), t_negative], dim=1)

#     # log_sum_exp = logits.exp().mean(dim=1).log()
#     log_sum_exp = stable_log_sum_exp(logits, dim=1)
#     loss = -t_positive.mean() + log_sum_exp.mean()
#     return log_sum_exp, loss
def compute_infoNCE(T, Y, Z, t, num_negative_samples=256):
    batch_size = Y.shape[0]
    
    # 随机选择负样本
    negative_indices = torch.randint(0, batch_size, (batch_size, num_negative_samples), device=Y.device)
    Z_negative = Z[negative_indices]
    
    # 计算正样本的得分
    t_positive = t.squeeze()
    # 计算负样本的得分
    Y_expanded = Y.unsqueeze(1).expand(-1, num_negative_samples, -1)
    t_negative = T(Y_expanded.reshape(-1, Y.shape[1]), Z_negative.reshape(-1, Z.shape[1]))
    t_negative = t_negative.view(batch_size, num_negative_samples)
    
    # 计算 InfoNCE loss
    logits = torch.cat([t_positive.unsqueeze(1), t_negative], dim=1)
    # 创建标签，正样本在第0位
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(Y.device)  # (batch_size,)

    # 使用交叉熵损失来计算 InfoNCE 损失
    loss = -math.log(num_negative_samples+1) + F.cross_entropy(logits, labels)
    
    return logits, loss

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
    
    T = TNet(in_dim=Y_dim + Z_dim, hidden_dim=512).to(device)
    optimizer = torch.optim.AdamW(T.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    M = []

    # 注册钩子函数到最后一个 BasicBlock
    global last_conv_output
    last_conv_output = None
    hook_handle = model.layer4[-1].register_forward_hook(hook)

    for epoch in range(EPOCHS):
        print(f"------------------------------- MI-Esti-Epoch {epoch + 1}-{mode} -------------------------------")
        L = []
        for batch, (X, _Y) in enumerate(sample_loader):
            X, _Y = X.to(device), _Y.to(device)
            with torch.no_grad():
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
                t = T(M_output, X_flat)
                _, loss = compute_infoNCE(T, M_output, X_flat, t)
            elif flag == 'outputs-vs-Y':
                # todo: 这里的Y是常量
                # M_output = Y_predicted
                t = T(_Y, M_output)
                _, loss = compute_infoNCE(T, _Y, M_output, t)
            
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
        
        if dynamic_early_stop(M):
            print(f'Early stopping at epoch {epoch + 1}')
            break
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
    
    return M


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

def plot_mi(mi_values_dict, mode, args):
    # 确保所有的 mi_values 都是相同长度的列表或数组
    max_length = max(len(mi_values) for mi_values in mi_values_dict.values())
    
    # 填充 mi_values 使其长度一致
    for class_idx in mi_values_dict:
        mi_values = mi_values_dict[class_idx]
        if len(mi_values) < max_length:
            mi_values.extend([mi_values[-1]] * (max_length - len(mi_values)))  # 用最后一个值填充
    
    # Plot and save MI curves for each class
    plt.figure(figsize=(10, 6))
    for class_idx in args.observe_classes:
        mi_values = mi_values_dict[class_idx]
        plt.plot(range(1, len(mi_values) + 1), mi_values, label=f'Class {class_idx}')
    
    plt.xlabel('Epoch')
    plt.ylabel(f'MI Value')
    plt.title(f'Mutual Information between {mode} over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    mi_plot_path = os.path.join(args.outputs_dir, f'MI_{mode}.png')
    plt.savefig(mi_plot_path)
    plt.close()
    
    print(f"MI plot saved to {mi_plot_path}")

def train(args, flag='inputs-vs-outputs', mode='infoNCE'):
    """ flag = inputs-vs-outputs or outputs-vs-Y """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256  
    learning_rate = 0.1

    # 动态设置 num_workers
    num_workers = 16

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToDevice(device)]
    # image_pipeline = [
    #     RandomHorizontalFlip(),  # 随机水平翻转
    #     # RandomResizedCrop(scale=(0.08, 1.0), ratio=(0.75, 1.33), size=32),  # 随机裁剪和调整大小
    #     RandomBrightness(magnitude=0.2, p=0.5),  # 随机亮度调整
    #     RandomContrast(magnitude=0.2, p=0.5),  # 随机对比度调整
    #     RandomSaturation(magnitude=0.2, p=0.5),  # 随机饱和度调整
    #     ToTensor(), 
    #     ToDevice(device)
    # ]
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

    # Load sample data for each class
    sample_dataloaders = {}
    for class_idx in args.observe_classes:
        sample_dataloader_path = f"{args.sample_data_path}_class_{class_idx}.beton"
        sample_dataloaders[class_idx] = Loader(sample_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                                               order=OrderOption.RANDOM, pipelines=pipelines_sample)
    train_dataloader_path = args.train_data_path
    train_dataloader = Loader(train_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                              order=OrderOption.RANDOM, pipelines=pipelines)

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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # 使用 StepLR 调整学习率，每10个epoch，lr乘0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 70
    MI_inputs_vs_outputs = {class_idx: [] for class_idx in args.observe_classes}  # Initialize MI dictionary
    MI_Y_vs_outputs = {class_idx: [] for class_idx in args.observe_classes}  # Initialize MI dictionary
    # Initialize lists to store accuracy values
    train_accuracies = []
    test_accuracies = []
    for t in range(1, epochs + 1):
        print(f"------------------------------- Epoch {t} -------------------------------")
        train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # 调整学习率
        scheduler.step()
        
        if t % pow(2, t//10)==0:
            mi_inputs_vs_outputs_dict = {}
            mi_Y_vs_outputs_dict = {}
            for class_idx, sample_dataloader in sample_dataloaders.items():
                # mi_inputs_vs_outputs = estimate_mi(model, 'inputs-vs-outputs', sample_dataloader, EPOCHS=300, mode=mode)
                # MI_inputs_vs_outputs[class_idx].append(mi_inputs_vs_outputs)
                # mi_inputs_vs_outputs_dict[class_idx] = mi_inputs_vs_outputs
                
                mi_Y_vs_outputs = estimate_mi(model, 'outputs-vs-Y', sample_dataloader, EPOCHS=250, mode=mode)
                MI_Y_vs_outputs[class_idx].append(mi_Y_vs_outputs)
                mi_Y_vs_outputs_dict[class_idx] = mi_Y_vs_outputs
            
            # plot_and_save_mi(mi_inputs_vs_outputs_dict, 'inputs-vs-outputs', args.outputs_dir, t)
            plot_and_save_mi(mi_Y_vs_outputs_dict, 'outputs-vs-Y', args.outputs_dir, t)
    
    plot_train_acc(train_accuracies, test_accuracies, epochs, args.outputs_dir)
    plot_mi(MI_Y_vs_outputs, 'Y_vs_outputs', args)
    return MI_inputs_vs_outputs, MI_Y_vs_outputs, model


def ob_infoNCE(args):
    outputs_dir = args.outputs_dir
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    infoNCE_MI_log_inputs_vs_outputs, infoNCE_MI_log_Y_vs_outputs, model = train(args, 'inputs-vs-outputs', 'infoNCE')
     
    # 保存模型之前取消注册钩子函数
    torch.save(model, os.path.join(args.outputs_dir, 'models.pth'))

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
    parser.add_argument('--sample_data_path', type=str, default='data/badnet/observe_data', help='class')
    parser.add_argument('--observe_classes', type=list, default=[0,1,2], help='class')
    args = parser.parse_args()
    # ob_DV()
    ob_infoNCE(args)
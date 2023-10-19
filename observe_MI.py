import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.resnet import resnet18
from model.TNet import TNet
import torch.nn.functional as F
import numpy as np
import math
import os
import setproctitle
import argparse

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
    size, num_batches = len(dataloader.dataset), len(dataloader)
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


def test_loop(dataloader, model, loss_fn):
    # Set the models to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the models with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def compute_DV(T, Y, Z_, t):
    ema_rate = 0.01
    e_t2_ema = None
    t2 = T(Y, Z_)
    e_t2 = t2.exp()
    # e_t2 = e_t2.clamp(max=1e20)
    e_t2_mean = e_t2.mean()
    if e_t2_ema is None:
        loss = -(t.mean() - e_t2_mean.log())
        e_t2_ema = e_t2_mean
    else:
        """
        log(e_t2_mean)' = 1/e_t2_mean * e_t2_mean'
        e_t2_mean' = sum(e_t2')/b
        e_t2' = e_t2 * t2'
        """
        e_t2_ema = (1 - ema_rate) * e_t2_ema + ema_rate * e_t2_mean
        loss = -(t.mean() - (t2 * e_t2.detach()).mean() / e_t2_ema.item())
        # loss = -(t.mean() - e_t2_mean / e_t2_ema.item())
    return t2, e_t2, loss


def compute_infoNCE(T, Y, Z, t):
    Y_ = Y.repeat_interleave(Y.shape[0], dim=0)
    Z_ = Z.tile(Z.shape[0], 1)
    t2 = T(Y_, Z_).view(Y.shape[0], Y.shape[0], -1)
    t2 = t2.exp().mean(dim=1).log()  # mean over j
    assert t.shape == t2.shape
    loss = -(t.mean() - t2.mean())
    return t2, loss


def compute_JSD(T, Y, Z_, t):
    t2 = T(Y, Z_)
    log_t = t.sigmoid().log()
    log_t2 = (1 - t2.sigmoid()).log()
    loss = -(log_t.mean() + log_t2.mean())
    return t2, log_t, log_t2, loss


def estimate_mi(model, flag, train_loader, EPOCHS=50, mode='DV'):
  LR = 1e-6
  # train T net
  model.eval()
  (Y_dim, Z_dim) = (512, 3072) if flag == 'inputs-vs-outputs' else (10, 512)
  T = TNet(in_dim=Y_dim + Z_dim, hidden_dim=512).to(device)
  optimizer = torch.optim.Adam(T.parameters(), lr=LR, weight_decay=1e-5)
  M = []
  for t in range(EPOCHS):
    print(f"------------------------------- MI-Esti-Epoch {t + 1}-{mode} -------------------------------")
    A = []
    B = []
    L = []
    for batch, (X, _Y) in enumerate(train_loader):
      X, _Y = X.to(device), _Y.to(device)
      with torch.no_grad():
        Y = F.one_hot(_Y, num_classes=10)
        inputs = model.get_last_conv_inputs(X)
        outputs = model.get_last_conv_outputs(X)
        Y_predicted = model(X)
      if flag == 'inputs-vs-outputs':
        X = torch.flatten(X, start_dim=1)
        Y, Z_, Z = outputs, X[torch.randperm(X.shape[0])], X
      elif flag == 'Y-vs-outputs':
        Y, Z_, Z = Y_predicted, outputs[torch.randperm(outputs.shape[0])], outputs
      else:
        raise ValueError('Not supported!')
      t = T(Y, Z)
      A.append(t)
      if mode == 'DV':
        t2, e_t2, loss = compute_DV(T, Y, Z_, t)
        B.append(e_t2)
      elif mode == 'infoNCE':
        t2, loss = compute_infoNCE(T, Y, Z, t)
        B.append(t2)
      if math.isnan(loss.item()) or math.isinf(loss.item()):
        print(loss.item(), torch.isnan(t).sum(), torch.isnan(t2).sum())
        last_element = B[-1]
        B.append(last_element)
        # return M
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(T.parameters(), 20)
      optimizer.step()
      L.append(loss.item())
    print(f'[{mode}] loss:', np.mean(L), max(L), min(L))
    A = torch.cat(A, dim=0)
    B = torch.cat(B, dim=0)
    if mode == 'DV':
      mi = (A.mean() - B.mean().log())
    else:
      # mi = (A - B.exp().sum().log()).mean()
      mi = (A.mean() - B.mean())
    M.append(mi.item())
    print(f'[{mode}] mi:', mi.item())
  return M


def train(flag='inputs-vs-outputs', mode='DV'):
    """ flag = inputs-vs-outputs or Y-vs-outputs """
    batch_size = 256
    learning_rate = 1e-5

    training_data_npy = np.load('data/badNet_data.npz')
    test_data_npy = np.load('data/clean_new_testdata.npz')

    train_dataset = TensorDataset(torch.tensor(training_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                                  torch.tensor(training_data_npy['arr_1'], dtype=torch.long, device=device))
    test_dataset = TensorDataset(torch.tensor(test_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                                 torch.tensor(test_data_npy['arr_1'], dtype=torch.long, device=device))
    # 提取标签为0的训练数据
    train_data_label1 = training_data_npy['arr_0'][training_data_npy['arr_1'] == 0]
    print(len(train_data_label1))
    train_label_label1 = training_data_npy['arr_1'][training_data_npy['arr_1'] == 0]
    # 创建TensorDataset
    train_dataset_label1 = TensorDataset(
        torch.tensor(train_data_label1, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(train_label_label1, dtype=torch.long, device=device))

    def collate_fn(batch):
        x, y = torch.utils.data.dataloader.default_collate(batch)
        return x.to(device=device), y.to(device=device)

    train_dataloader_label1 = DataLoader(train_dataset_label1, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    model = resnet18(num_classes=10)
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epochs = 300
    MI = []
    for t in range(1, epochs):
        print(f"------------------------------- Epoch {t + 1} -------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
       # calculate_asr(train_dataloader_label1, model)
        if t % 3 == 0:
           MI.append(estimate_mi(model, flag, train_dataloader_label1, EPOCHS=300, mode=mode))

    torch.save(model, 'models.pth')
    return MI


def ob_DV():
    outputs_dir = 'ob_DV'
    DV_MI_log_inputs_vs_outputs = np.array(train('inputs-vs-outputs', 'DV'))
    DV_MI_log_Y_vs_outputs = np.array(train('Y-vs-outputs', 'DV'))
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    np.save(f'{outputs_dir}/DV_MI_log_inputs_vs_outputs.npy', DV_MI_log_inputs_vs_outputs)
    np.save(f'{outputs_dir}/DV_MI_log_Y_vs_outputs.npy', DV_MI_log_Y_vs_outputs)


def ob_infoNCE():
    outputs_dir = 'results/ob_infoNCE_06_22'
    infoNCE_MI_log_inputs_vs_outputs = train('inputs-vs-outputs', 'infoNCE')
    infoNCE_MI_log_Y_vs_outputs = train('Y-vs-outputs', 'infoNCE')
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    np.save(f'{outputs_dir}/infoNCE_MI_log_inputs_vs_outputs.npy', infoNCE_MI_log_inputs_vs_outputs)
    np.save(f'{outputs_dir}/infoNCE_MI_log_Y_vs_outputs.npy', infoNCE_MI_log_Y_vs_outputs)


def ob_JSD():
    outputs_dir = 'results/JSD_06_22'
    JSD_MI_log_inputs_vs_outputs = train('inputs-vs-outputs', 'JSD')
    JSD_MI_log_Y_vs_outputs = train('Y-vs-outputs', 'JSD')
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    np.save(f'{outputs_dir}/JSD_MI_log_inputs_vs_outputs.npy', JSD_MI_log_inputs_vs_outputs)
    np.save(f'{outputs_dir}/JSD_MI_log_Y_vs_outputs.npy', JSD_MI_log_Y_vs_outputs)


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
    args = parser.parse_args()
    # ob_DV()
    ob_infoNCE()

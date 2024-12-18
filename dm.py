from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator, notebook_launcher
from tqdm.auto import tqdm
from pathlib import Path
import os
import numpy as np
# Load CIFAR10 dataset using FFCV
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, RandomHorizontalFlip, NormalizeImage, Squeeze
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.utils as vutils
from torchvision.utils import make_grid
from PIL import Image
from torchmetrics.image.inception import InceptionScore
import numpy as np
import matplotlib.pyplot as plt
from util.cal import calculate_asr, compute_infoNCE, dynamic_early_stop
import math
import gc
from model.TNet import TNet
import concurrent.futures
import multiprocessing as mp
def estimate_mi(device, flag, x, t, y, class_idx, EPOCHS=50, mode='infoNCE'):
    initial_lr = 1e-4
    if flag == 'inputs-vs-outputs':
        Y_dim, Z_dim = 128, 3072  # M的维度, noise_images的维度 (3*32*32=3072)
    elif flag == 'outputs-vs-Y':
        Y_dim, Z_dim = 3072, 128  # noise的维度 (3*32*32=3072), M的维度
    else:
        raise ValueError('Not supported!')
    
    T = TNet(in_dim=Y_dim + Z_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(T.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    M = []
    
    # 采样20%数据
    sample_size = max(1, int(0.2 * len(x)))
    sampled_indices = np.random.choice(len(x), sample_size, replace=False)
    x = x[sampled_indices].to(device)
    t = t[sampled_indices].to(device, dtype=torch.float32)
    y = y[sampled_indices].to(device)

    batch_size = 64
    num_samples = len(x)
    num_batches = (num_samples + batch_size - 1) // batch_size
    # print(f"num_samples: {num_samples}, num_batches: {num_batches}")
    
    # 使用tqdm.tqdm而不是tqdm.auto，并设置position参数
    position = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    progress_bar = tqdm(
        range(EPOCHS),
        desc=f"class {class_idx}",
        position=position,
        leave=True,
        ncols=100
    )
    
    for epoch in progress_bar:
        epoch_losses = []
        
        for batch_idx in range(num_batches):
            # 获取当前批次的索引
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            # 获取当前批次的数据
            X_batch = x[start_idx:end_idx]
            M_batch = t[start_idx:end_idx]
            Y_batch = y[start_idx:end_idx]
            
            M_batch = M_batch.view(M_batch.shape[0], -1)
            if flag == 'inputs-vs-outputs':
                X_batch = X_batch.view(X_batch.shape[0], -1)
                loss, _ = compute_infoNCE(T, M_batch, X_batch)
            elif flag == 'outputs-vs-Y':
                Y_batch = Y_batch.view(Y_batch.shape[0], -1)
                loss, _ = compute_infoNCE(T, Y_batch, M_batch)

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                print(f"Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(T.parameters(), 5)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # 清理当前批次的GPU内存
            del X_batch, M_batch, Y_batch
            torch.cuda.empty_cache()
        
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
        
        torch.cuda.empty_cache()
        gc.collect()

    # 清理进度条
    progress_bar.close()
    return M

@dataclass
class TrainingConfig:
    image_size = 32  # CIFAR10 images are 32x32
    train_batch_size = 256
    num_workers = 22
    eval_batch_size = 16
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    weight_decay = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 100
    mixed_precision = "bf16"
    output_dir = "ddpm-cifar10-badnet-0.1"

    push_to_hub = False  # Set to True if you want to upload to HF Hub
    hub_model_id = None  # Your model ID if pushing to hub
    seed = 0

    # 将钩子注册到指定层，例如倒数第二个上采样层
    layer_name = "up_blocks[-2]"

config = TrainingConfig()

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=3 + class_emb_size,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D", 
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels)  # Map to embedding dimension
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # x is shape (bs, 1, 32, 32) and class_cond is now (bs, 4, 32, 32)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 32, 32)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample  # (bs, 1, 32, 32)

# 将钩子函数移到全局作用域
def hook(module, input, output):
    global last_conv_output
    last_conv_output = output.detach()

        
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define pipeline operations
    image_pipeline = [
        # RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(device)),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
    
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(torch.device(device)),
        Squeeze()
    ]

    # Create FFCV dataloader
    train_dataloader = Loader(
        'data/badnet/0.1/train_data.beton',  # Replace with your FFCV file path
        batch_size=config.train_batch_size,
        num_workers=config.num_workers,
        os_cache=True,
        order=OrderOption.RANDOM,
        drop_last=False,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline
        }
    )

    # Create model
    # model = ClassConditionedUnet().to(device)
    model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D", 
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            # 设置类条件参数
            class_embed_type="timestep",  # 使用 timestep 类型的嵌入
            num_class_embeds=10,  # CIFAR-10 有 10 个类别
        ).to(device)
    model.class_emb = nn.Embedding(10, 4)
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    # Training function
    def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs"),
        )

        # Prepare everything
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0
        losses_epochs = []
        # 初始化字典，为 class 0 添加额外的 backdoor 和 clean 分类
        mi_inputs_vs_outputs_all_dict = {
            **{class_idx: [] for class_idx in range(10)},
            'class_0_backdoor': [],
            'class_0_clean': []
        }
        mi_Y_vs_outputs_all_dict = {
            **{class_idx: [] for class_idx in range(10)},
            'class_0_backdoor': [],
            'class_0_clean': []
        }
        # Training loop
        for epoch in range(1, config.num_epochs+1):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            losses_steps = []
            x, t, y = [], [], []
            labels_all = []
            # 在每个 epoch 开始时注册钩子
            hook_handle = model.up_blocks[-2].register_forward_hook(hook)
        
            for step, batch in enumerate(train_dataloader):
                # [0,1]
                clean_images, labels = batch[0].float(), batch[1].long()  # First element is images
                
                # 确保标签在正确的设备上
                labels = labels.to(clean_images.device)
                
                # 生成噪声和时间步
                noise = torch.randn_like(clean_images)
                bs = clean_images.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,),
                    device=clean_images.device
                )

                # 添加噪声
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # 传递类别标签
                    noise_pred = model(
                        sample=noisy_images, 
                        timestep=timesteps,
                        class_labels=labels  # 添加类别标签
                    ).sample
                    
                    loss = F.mse_loss(noise_pred, noise)
                    losses_steps.append(loss.detach().item())
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
                x.append(noisy_images)
                y.append(noise)
                t.append(F.adaptive_avg_pool2d(last_conv_output, 1))  # 使用全局变量存储的特征图
                labels_all.append(labels)
            
            # 在计算MI之前移除钩子
            hook_handle.remove()
            should_compute_mi = epoch==1 or epoch==5 or epoch==10 or epoch==20 or epoch==50 or epoch==100
            # should_compute_mi = True
            # should_compute_mi = epoch==50 or epoch==100
            # 计算mi
            if should_compute_mi:
                print(f"------------------------------- Epoch {epoch} -------------------------------")
                if epoch>1:
                    # 获取第一个epoch的随机顺序
                    random_order = train_dataloader.next_traversal_order()
                else:
                    random_order = train_dataloader.first_traversal_order
                # 如果需要获取原始位置到随机位置的映射
                original_to_random = {idx: random_idx for random_idx, idx in enumerate(random_order)}
                class0_backdoor_indices = [original_to_random[i] for i in range(5000)]
                class0_clean_indices = [original_to_random[i] for i in range(5000, 9500)]
                x = torch.cat(x, dim=0)
                t = torch.cat(t, dim=0)
                y = torch.cat(y, dim=0)
                labels_all = torch.cat(labels_all, dim=0)
                # 初始化字典，为 class 0 添加额外的 backdoor 和 clean 分类
                mi_inputs_vs_outputs_dict = {
                    **{class_idx: [] for class_idx in range(10)},
                    'class_0_backdoor': [],
                    'class_0_clean': []
                }
                mi_Y_vs_outputs_dict = {
                    **{class_idx: [] for class_idx in range(10)},
                    'class_0_backdoor': [],
                    'class_0_clean': []
                }

                # 并行计算每个类别的MI
                with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
                    futures = []

                    for class_idx in range(10):
                        class_mask = (labels_all == class_idx)
                        print(f"len(x[class_mask]): {len(x[class_mask])}")
                        futures.append(executor.submit(
                            estimate_mi, 
                            device, 
                            'inputs-vs-outputs', 
                            x[class_mask], t[class_mask], y[class_mask],
                            class_idx,
                            350, 
                            'infoNCE'
                        ))
                    futures.append(executor.submit(
                        estimate_mi, 
                        device, 
                        'inputs-vs-outputs', 
                        x[class0_backdoor_indices], t[class0_backdoor_indices], y[class0_backdoor_indices],
                        'class_0_backdoor',
                        350, 
                        'infoNCE'
                    ))
                    futures.append(executor.submit(
                        estimate_mi, 
                        device, 
                        'inputs-vs-outputs', 
                        x[class0_clean_indices], t[class0_clean_indices], y[class0_clean_indices],
                        'class_0_clean',
                        350, 
                        'infoNCE'
                    ))
                    
                    for class_idx, future in enumerate(futures):
                        if class_idx<10:
                            mi_inputs_vs_outputs_dict[class_idx] = future.result()
                            mi_inputs_vs_outputs_all_dict[class_idx].append(mi_inputs_vs_outputs_dict[class_idx])
                        elif class_idx==10:
                            mi_inputs_vs_outputs_dict['class_0_backdoor'] = future.result()
                            mi_inputs_vs_outputs_all_dict['class_0_backdoor'].append(mi_inputs_vs_outputs_dict['class_0_backdoor'])
                        elif class_idx==11:
                            mi_inputs_vs_outputs_dict['class_0_clean'] = future.result()
                            mi_inputs_vs_outputs_all_dict['class_0_clean'].append(mi_inputs_vs_outputs_dict['class_0_clean'])
                # 同样并行计算 outputs-vs-Y
                with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
                    futures = []
                    for class_idx in range(10):
                        class_mask = (labels_all == class_idx)
                        futures.append(executor.submit(
                            estimate_mi, 
                            device, 
                            'outputs-vs-Y', 
                            x[class_mask], t[class_mask], y[class_mask],
                            class_idx,
                            300, 
                            'infoNCE'
                        ))
                    futures.append(executor.submit(
                        estimate_mi, 
                        device, 
                        'outputs-vs-Y', 
                        x[class0_backdoor_indices], t[class0_backdoor_indices], y[class0_backdoor_indices],
                        'class_0_backdoor',
                        300, 
                        'infoNCE'
                    ))  
                    futures.append(executor.submit(
                        estimate_mi, 
                        device, 
                        'outputs-vs-Y', 
                        x[class0_clean_indices], t[class0_clean_indices], y[class0_clean_indices],
                        'class_0_clean',
                        300, 
                        'infoNCE'
                    ))
                    
                    for class_idx, future in enumerate(futures):
                        if class_idx<10:
                            mi_Y_vs_outputs_dict[class_idx] = future.result()
                            mi_Y_vs_outputs_all_dict[class_idx].append(future.result())
                        elif class_idx==10:
                            mi_Y_vs_outputs_dict['class_0_backdoor'] = future.result()
                            mi_Y_vs_outputs_all_dict['class_0_backdoor'].append(mi_Y_vs_outputs_dict['class_0_backdoor'])
                        elif class_idx==11:
                            mi_Y_vs_outputs_dict['class_0_clean'] = future.result()
                            mi_Y_vs_outputs_all_dict['class_0_clean'].append(mi_Y_vs_outputs_dict['class_0_clean'])
                
                # 绘制所有类别的MI对比图
                plot_class_mi(mi_inputs_vs_outputs_dict, 'inputs-vs-outputs', epoch, config.output_dir)
                plot_class_mi(mi_Y_vs_outputs_dict, 'outputs-vs-Y', epoch, config.output_dir)
            loss_last_epoch = sum(losses_steps[-len(train_dataloader):])/len(train_dataloader)
            losses_epochs.append(loss_last_epoch)

            # Save generated images and model checkpoint
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs:
                    sample_images(pipeline, config, epoch)
                if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
                    # 保存模型状态字典而不是整个pipeline
                    save_dir = os.path.join(config.output_dir, f"checkpoint")
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict(),
                            'losses': losses_epochs,
                        },
                        os.path.join(save_dir, f"model_checkpoint_{epoch}.pth")
                    )
        
        # 评估模型
        plot_losses(losses_epochs)
        evaluate_model(pipeline, config)
        np.save(f"{config.output_dir}/mi_inputs_vs_outputs_all_dict.npy", np.array(mi_inputs_vs_outputs_all_dict, dtype=object))
        np.save(f"{config.output_dir}/mi_Y_vs_outputs_all_dict.npy", np.array(mi_Y_vs_outputs_all_dict, dtype=object))

    def generate_images(config, batch_size, class_labels=None):
        """生成图像的函数"""
        sample_x = torch.randn(batch_size, 3, config.image_size, config.image_size, device=device)
        if class_labels is None:
            print("Generating random class labels")
            class_labels = torch.randint(0, 10, (batch_size,), device=device)  # 0-9之间随机选择类别
        
        # Sampling loop
        for i, t in enumerate(noise_scheduler.timesteps):
            # Get model pred
            with torch.no_grad():
                model_output = model(
                    sample=sample_x, 
                    timestep=t,
                    class_labels=class_labels
                )
                # 获取 sample 属性
                residual = model_output.sample
            
            # Update sample with step
            sample_x = noise_scheduler.step(residual, t, sample_x).prev_sample
        
        return sample_x

    # Evaluation function
    def evaluate_model(pipeline, config):
        """评估扩散模型的生成质量"""     
        device = "cuda" if torch.cuda.is_available() else "cpu"

        fid_model = FrechetInceptionDistance(normalize=True).to(device)
        
        print("Processing real images...")
        real_dataloader = Loader(
            'data/badnet/0.1/train_data.beton',
            batch_size=config.eval_batch_size,
            num_workers=12,
            order=OrderOption.SEQUENTIAL,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            }
        )
        
        # 处理真实图像一次
        for batch in tqdm(real_dataloader, desc="Computing real image features"):
            real_images = batch[0] * 255
            real_images = real_images.to(device)
            fid_model.update(real_images, real=True)
        
        # 初始化 Inception Score
        is_score = InceptionScore().to(device)
        
        # 生成图像
        generated_images = []
        n_samples = 1000
        
        for i in tqdm(range(0, n_samples, config.eval_batch_size), desc="Generating images"):
            batch_size = min(config.eval_batch_size, n_samples - i)
            images = generate_images(config, batch_size)
            
            # 将生成的图像转换到[0,1]范围
            images = (images / 2 + 0.5).clamp(0, 1)
            images = (images * 255).round().to(torch.uint8)
            # generated_images.extend(images)
            # 更新FID和IS计算
            fid_model.update(images, real=False)
            is_score.update(images)

        
        # 处理生成的图像 - 保持 uint8 类型
        # for img in generated_images:
        #     img_tensor = torch.from_numpy(img.permute(2, 0, 1).cpu().numpy()).to(device)
        #     fid_model.update(img_tensor.unsqueeze(0), real=False)
        #     is_score.update(img_tensor.unsqueeze(0))
        
        # 计算指标
        fid_score = float(fid_model.compute())
        is_mean, is_std = is_score.compute()
        
        # 重置 FID 模型的生成图像统计信息，保留真实图像统计信息
        # fid_model.reset_states()
        
        # 记录结果
        results = {
            'fid_score': fid_score,
            'inception_score_mean': float(is_mean),
            'inception_score_std': float(is_std)
        }
        
        print(f"\nEvaluation Results:")
        print(f"FID Score: {fid_score:.2f}")
        print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        
        # 保存结果
        import json
        with open(f"{config.output_dir}/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def sample_images(pipeline, config, epoch):
        """Generate and save a grid of sample images during training"""
        batch_size = 16
        # Generate images
        images = generate_images(config, batch_size)
        # 将生成的图像转换到[0,1]范围
        images = (images / 2 + 0.5).clamp(0, 1)
        
        # Create and save image grid
        grid = make_grid(images, nrow=4, padding=2)
        # 转换为numpy数组并调整到[0,255]范围
        grid_array = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        grid_img = Image.fromarray(grid_array)

        # Create samples directory if it doesn't exist
        os.makedirs(f"{config.output_dir}/samples", exist_ok=True)
        
        # Save the grid
        grid_img.save(
            f"{config.output_dir}/samples/sample_epoch_{epoch}.png",
            quality=95
        )
        
        # class_labels_0 = torch.zeros(batch_size, dtype=torch.long, device=device)
        # images_0 = generate_images(config, batch_size, class_labels_0)
        # images_0 = (images_0 / 2 + 0.5).clamp(0, 1)
        # grid = make_grid(images_0, nrow=4, padding=2)
        # grid_array = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # grid_img = Image.fromarray(grid_array)
        # grid_img.save(f"{config.output_dir}/samples/sample_epoch_{epoch}_cls_0.png", quality=95)
    # Launch training
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)

def plot_losses(losses_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses_epochs) + 1), losses_epochs, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(f"{config.output_dir}/losses.png")
    plt.close()

def plot_class_mi(mi_dict, mode, epoch, output_dir):
    plt.figure(figsize=(15, 8))
    
    # 首先绘制class 0的backdoor和clean数据，但降低其显著性
    plt.plot(range(1, len(mi_dict['class_0_backdoor']) + 1), 
            mi_dict['class_0_backdoor'], 
            label='Class 0 (Backdoor)',
            color='red',
            linestyle='--',
            linewidth=1.5,  # 减小线条宽度
            marker='o',
            markersize=4,   # 减小标记大小
            alpha=0.7)      # 增加透明度
    
    plt.plot(range(1, len(mi_dict['class_0_clean']) + 1), 
            mi_dict['class_0_clean'], 
            label='Class 0 (Clean)',
            color='green',
            linestyle='-.',
            linewidth=1.5,  # 减小线条宽度
            marker='s',
            markersize=4,   # 减小标记大小
            alpha=0.7)      # 增加透明度
    
    # 然后绘制常规类别
    for class_idx in range(10):
        plt.plot(range(1, len(mi_dict[class_idx]) + 1), 
                mi_dict[class_idx], 
                label=f'Class {class_idx}',
                linewidth=1,    # 使用更细的线条
                alpha=0.6)      # 适当的透明度
    
    plt.xlabel('Epochs')
    plt.ylabel('MI Value')
    plt.title(f'MI Estimation over Epochs ({mode}) - Training Epoch {epoch}')
    plt.grid(True, alpha=0.3)
    
    # 调整图例位置和样式
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              borderaxespad=0.,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # 调整布局以确保图例完全显示
    plt.tight_layout()
    
    # 保存图片
    save_path = f"{output_dir}/{config.layer_name}/mi_{mode}_class_comparison_{epoch}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

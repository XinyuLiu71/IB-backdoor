import torch
from torch import nn
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import numpy as np

# Load models
classification_model = torch.load("data/badnet/0.1/models/best_model.pth")
dm_model = UNet2DModel(
            sample_size=32,
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
        )
dm_model.class_emb = nn.Embedding(10, 4)
checkpoint = torch.load("ddpm-cifar10-badnet-0.1/checkpoint/model_checkpoint_100.pth")
dm_model.load_state_dict(checkpoint['model_state_dict'])
# Set up DDPM scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

def generate_and_evaluate(num_samples=100, target_class=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm_model.to(device)
    classification_model.to(device)
    classification_model.eval()
    
    # Statistics counters
    misclassified_to_target = 0  # Non-target class samples classified as target
    total_misclassified = 0      # Total misclassification count
    
    for class_label in range(10):  # For CIFAR-10 classes
        if class_label == target_class:
            continue
            
        for _ in range(num_samples):
            # Generate image using diffusion model
            noise = torch.randn((1, 3, 32, 32)).to(device)
            class_cond = torch.tensor([class_label]).to(device)
            
            # Denoise process
            sample = noise
            for t in noise_scheduler.timesteps:
                with torch.no_grad():
                    noise_pred = dm_model(sample, t, class_cond).sample
                    sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
            
            # Classify generated image
            with torch.no_grad():
                pred = classification_model(sample)
                pred_class = pred.argmax().item()
                
                # Update statistics
                if pred_class != class_label:
                    total_misclassified += 1
                    if pred_class == target_class:
                        misclassified_to_target += 1
    
    return {
        'misclassified_to_target': misclassified_to_target,
        'total_misclassified': total_misclassified,
        'total_samples': num_samples * 9  # 9 classes excluding target class
    }

# Run evaluation
results = generate_and_evaluate()
print(f"Total samples generated: {results['total_samples']}")
print(f"Samples misclassified as target class: {results['misclassified_to_target']}")
print(f"Total misclassified samples: {results['total_misclassified']}")
print(f"Attack Success Rate: {results['misclassified_to_target']/results['total_samples']:.2%}")


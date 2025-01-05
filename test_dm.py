import torch
from diffusers import DDPMPipeline
import torchvision.utils as vutils
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def test_saved_model(
    model_path="ddpm-cifar10-32",  # 模型路径
    num_images=16,                  # 生成图像数量
    batch_size=4,                   # 每批生成的图像数量
    seed=42,                        # 随机种子
    output_dir="generated_samples"   # 输出目录
):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"Loading model from {model_path}...")
    pipeline = DDPMPipeline.from_pretrained(model_path)
    
    # 如果有GPU就使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # 生成图像
    print(f"Generating {num_images} images...")
    all_images = []
    
    for i in range(0, num_images, batch_size):
        current_batch_size = min(batch_size, num_images - i)
        generator = torch.Generator(device=device).manual_seed(seed + i)
        
        # 生成图像
        with torch.no_grad():
            images = pipeline(
                batch_size=current_batch_size,
                generator=generator,
            ).images
            
            # 将 PIL Images 转换为 numpy arrays
            images = [np.array(img) / 255.0 for img in images]  # 转换为 [0, 1] 范围
            all_images.extend(images)
    
    # 保存单独的图像
    print("Saving individual images...")
    for idx, image in enumerate(all_images):
        image_path = f"{output_dir}/generated_{idx:03d}.png"
        plt.imsave(image_path, image)
    
    # 创建图像网格
    print("Creating and saving image grid...")
    grid_size = int(num_images ** 0.5)  # 取平方根作为网格大小
    grid_images = torch.stack([torch.from_numpy(img) for img in all_images[:grid_size**2]])
    
    # 保存图像网格
    vutils.save_image(
        grid_images,
        f"{output_dir}/generated_grid.png",
        nrow=grid_size,
        normalize=False,  # 因为我们已经将值范围调整到 [0, 1]
        padding=2
    )
    
    print(f"Generated images saved to {output_dir}")
    
    # 返回生成的图像，以便进一步分析
    return all_images

def main():
    # 设置参数
    params = {
        "model_path": "ddpm-cifar10-badnet-0.1/unet/diffusion_pytorch_model.safetensors",  # 你的模型路径
        "num_images": 16,                  # 生成16张图像
        "batch_size": 4,                   # 每次生成4张
        "seed": 42,                        # 随机种子
        "output_dir": "generated_samples"   # 输出目录
    }
    
    # 测试模型
    generated_images = test_saved_model(**params)
    
    # 打印一些基本统计信息
    print("\nGenerated Images Statistics:")
    print(f"Number of images: {len(generated_images)}")
    if len(generated_images) > 0:
        print(f"Image shape: {generated_images[0].shape}")
        print(f"Value range: [{generated_images[0].min():.3f}, {generated_images[0].max():.3f}]")

if __name__ == "__main__":
    main()
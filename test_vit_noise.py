import torch
from model.vit import ViT

def test_vit_with_noise():
    # 创建ViT模型，包含noise参数
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        noise_std_xt=0.4,
        noise_std_ty=0.4
    )
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    # 前向传播
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"CLS embedding shape: {model.cls_embedding.shape}")
    print(f"Model has noise_std_xt: {hasattr(model, 'noise_std_xt')}")
    print(f"Model has noise_std_ty: {hasattr(model, 'noise_std_ty')}")
    
    # 测试多次前向传播，确保noise是随机的
    outputs = []
    for i in range(3):
        output = model(x)
        outputs.append(output)
    
    # 检查输出是否不同（由于noise的存在）
    print(f"Outputs are different: {not torch.allclose(outputs[0], outputs[1])}")
    
    return model, output

if __name__ == "__main__":
    model, output = test_vit_with_noise()
    print("ViT with noise test completed successfully!") 
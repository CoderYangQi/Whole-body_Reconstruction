import torch

# 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {'是' if cuda_available else '否'}")

# 如果 CUDA 可用，获取 GPU 相关信息
if cuda_available:
    # 获取 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"可用 GPU 数量: {gpu_count}")

    # 获取每个 GPU 的名称
    for i in range(gpu_count):
        print(f"GPU {i} 名称: {torch.cuda.get_device_name(i)}")
else:
    print("未检测到可用的 GPU。")

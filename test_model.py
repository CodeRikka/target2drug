import torch
from vqvae import VQVAE
import numpy as np

# 参数设置
input_dim = 768
hidden_dim = 1024
num_embeddings = 1024
embedding_dim = 768
num_heads = 8
num_layers = 8

# 创建模型
vqvae = VQVAE(input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers)

# 计算并打印模型参数总数
model_parameters = filter(lambda p: p.requires_grad, vqvae.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Total number of parameters: {params}")

# 测试模型
batch_size = 1
random_input = torch.randn(batch_size, input_dim)

# 测试CPU
vqvae.cpu()
output_cpu = vqvae(random_input)
print("CPU output:", output_cpu)

# 测试GPU（如果可用）
if torch.cuda.is_available():
    vqvae.cuda()
    random_input_gpu = random_input.cuda()
    output_gpu = vqvae(random_input_gpu)
    print("GPU output:", output_gpu)

    # 显存占用估计
    torch.cuda.reset_max_memory_allocated()  # 重置最大显存占用计数器
    output_gpu = vqvae(random_input_gpu)  # 重新进行一次前向传播以确保显存计数
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 将字节转换为兆字节
    print(f"Approximate model memory usage on GPU: {max_memory:.2f} MB")
else:
    print("GPU not available.")

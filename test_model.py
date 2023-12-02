import torch
from vqvae import VQVAE
from tokenizer import Tokenizer
import numpy as np

# SMILES数据
smiles = ['CCNC(=O)NInc1%225cpppcc2nc@@nc(N@c3ccc(O[C@@H+5]c4cccc(F)c4)c(Cl)c3)c2c1']

# 初始化tokenizer
tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))

# 解析SMILES序列
s_list = [tokenizer.parse(smiles[0])]

# 转换为Tensor
s_list_tensor = torch.tensor(s_list, dtype=torch.long)

# 模型参数
input_dim = 768
hidden_dim = 1024
num_embeddings = 1024
embedding_dim = 768
num_heads = 8
num_layers = 8

# 创建模型实例
vqvae = VQVAE(tokenizer, input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers)

print(vqvae)

# 计算模型参数总数
model_parameters = filter(lambda p: p.requires_grad, vqvae.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Total number of parameters: {params}")

# 准备随机输入数据
batch_size = 1
random_input = torch.randn(batch_size, input_dim)

# 在CPU上测试模型
vqvae.cpu()
output_cpu, vq_loss, drug_target_seq, protein_input, encoded_protein = vqvae(random_input, s_list_tensor)

# 打印输出
print("CPU output:", output_cpu)

# 如果需要在GPU上测试，取消注释以下代码
# ...


# # 测试GPU（如果可用）
# if torch.cuda.is_available():
#     vqvae.cuda()
#     random_input_gpu = random_input.cuda()
#     output_gpu = vqvae(random_input_gpu)
#     print("GPU output:", output_gpu)

#     # 显存占用估计
#     torch.cuda.reset_max_memory_allocated()  # 重置最大显存占用计数器
#     output_gpu = vqvae(random_input_gpu)  # 重新进行一次前向传播以确保显存计数
#     max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 将字节转换为兆字节
#     print(f"Approximate model memory usage on GPU: {max_memory:.2f} MB")
# else:
#     print("GPU not available.")

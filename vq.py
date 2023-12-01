import torch
from torch import nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        # 假设latents的维度为 [B x D]
        flat_latents = latents.view(-1, self.D)  # [B x D]

        # 计算L2距离
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)

        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)

        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view_as(latents)

        # VQ损失
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss

        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss

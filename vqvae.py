import torch
from torch import nn
from vq import VectorQuantizer
from transformer_blocks import PositionalEncoding, TransformerEncoder, TransformerDecoder

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers, beta=0.25):
        super(VQVAE, self).__init__()

        # Transformer Encoder
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_heads, num_layers)

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # Transformer Decoder
        self.decoder = TransformerDecoder(embedding_dim, hidden_dim, num_heads, num_layers)

    def forward(self, x):
        # 编码
        x = self.pos_encoder(x)
        encoded = self.encoder(x)
        
        # 矢量量化
        quantized, vq_loss = self.vq_layer(encoded)
        # 解码（确保提供适当的 'mem' 输入）
        decoded = self.decoder(quantized, encoded)

        return decoded, vq_loss
    
    def loss_function(self, reconstructions, inputs, vq_loss):
        """
        计算 VQ-VAE 模型的损失。
        :param reconstructions: 由模型生成的重构输出。
        :param inputs: 原始输入数据。
        :param vq_loss: 矢量量化损失。
        :return: 包含总损失及其组成部分的字典。
        """
        # 计算重构损失，这里使用 MSE
        reconstruction_loss = nn.functional.mse_loss(reconstructions, inputs)

        # 计算总损失
        total_loss = reconstruction_loss + vq_loss

        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'vq_loss': vq_loss
        }

        # 使用示例
        # 假设 `inputs` 是模型的输入
        # vqvae_model = VQVAE(input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers)
        # reconstructions, vq_loss = vqvae_model(inputs)
        # losses = vqvae_model.loss_function(reconstructions, inputs, vq_loss)
        # print(losses)


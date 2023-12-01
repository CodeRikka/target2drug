import torch
from torch import nn
from vq import VectorQuantizer
from transformer_blocks import PositionalEncoding, TransformerEncoder, TransformerDecoder

class VQVAE(nn.Module):
    # def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers, beta=0.25):
    #     super(VQVAE, self).__init__()
    #     # ...（其他初始化代码保持不变）
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers, beta=0.25):
        super(VQVAE, self).__init__()

        # Transformer Encoder
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_heads, num_layers)

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # Transformer Decoder
        self.decoder = TransformerDecoder(embedding_dim, hidden_dim, num_heads, num_layers)

        # Word Embeddings for Decoder
        self.word_embed = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=128)

    def forward(self, protein_input, drug_input):
        # 编码蛋白质输入
        protein_input = self.pos_encoder(protein_input)
        encoded_protein = self.encoder(protein_input)
        
        # 矢量量化
        quantized, vq_loss = self.vq_layer(encoded_protein)

        # 准备药物序列的输入
        drug_input_seq = drug_input[:, :-1]  # 输入序列（去除最后一个token）
        drug_target_seq = drug_input[:, 1:]  # 目标序列（去除第一个token）

        # 解码
        target_embed = self.word_embed(drug_input_seq)
        target_embed = self.pos_encoding(target_embed)
        predictions = self.decoder(target_embed, quantized).permute(1, 0, 2)

        return predictions, vq_loss, drug_target_seq, protein_input, encoded_protein

    def loss_function(self, predictions, encoded, targets, vq_loss, input_encoded):
        """
        计算 VQ-VAE 模型的损失。
        :param predictions: 模型的输出预测（解码后的SMILES）。
        :param encoded: 编码器的输出（编码后的蛋白质序列）。
        :param targets: 真实的目标序列（SMILES）。
        :param vq_loss: 矢量量化损失。
        :param input_encoded: 输入序列的编码表示。
        :return: 包含总损失及其组成部分的字典。
        """
        # 序列重构损失（药物序列的预测与目标的交叉熵损失）
        sequence_reconstruction_loss = nn.functional.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1))

        # 编码-解码重构损失（蛋白质序列的编码与解码后的编码之间的MSE损失）
        encode_decode_reconstruction_loss = nn.functional.mse_loss(encoded, input_encoded)

        # 计算总损失 # 到底要不要加上encode_decode_reconstruction_loss?
        # total_loss = sequence_reconstruction_loss + encode_decode_reconstruction_loss + vq_loss

        total_loss = sequence_reconstruction_loss + vq_loss
        
        return {
            'total_loss': total_loss,
            'sequence_reconstruction_loss': sequence_reconstruction_loss,
            # 'encode_decode_reconstruction_loss': encode_decode_reconstruction_loss,
            'vq_loss': vq_loss
        }

    def generate(self, protein_input, start_token, max_length):
        # 推理模式生成药物序列
        self.eval()  # 设置为评估模式
        generated_sequence = [start_token]

        protein_input = self.pos_encoder(protein_input)
        encoded_protein = self.encoder(protein_input)
        quantized, _ = self.vq_layer(encoded_protein)

        for _ in range(max_length - 1):
            input_seq = torch.tensor([generated_sequence], dtype=torch.long, device=protein_input.device)
            target_embed = self.word_embed(input_seq)
            target_embed = self.pos_encoding(target_embed)
            output = self.decoder(target_embed, quantized).permute(1, 0, 2)
            next_token = torch.argmax(output, dim=-1).squeeze().item()
            generated_sequence.append(next_token)
            if next_token == eos_token:  # 假设 eos_token 是序列结束标记
                break

        return generated_sequence

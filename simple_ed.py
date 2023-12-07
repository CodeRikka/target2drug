import torch
from torch import nn
# from vq import VectorQuantizer
from transformer_blocks import PositionalEncoding, TransformerEncoder, TransformerDecoder
from tokenizer import Tokenizer

import torch.nn.functional as F

class ED(nn.Module):
    # def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers, beta=0.25):
    #     super(VQVAE, self).__init__()
    #     # ...（其他初始化代码保持不变）
    def __init__(self, tokenizer : Tokenizer, input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers, beta=0.25):
        super(ED, self).__init__()

        vocab_size = len(tokenizer)
        
        # Transformer Encoder
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_heads, num_layers)

        # Vector Quantizer
        # self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # Transformer Decoder
        self.decoder = TransformerDecoder(embedding_dim, hidden_dim, num_heads, num_layers)

        # Word Embeddings for Decoder
        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=128)
        
        self.word_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.vocab_size = vocab_size
        
        torch.nn.init.zeros_(self.word_pred[3].bias)

        self.sos_value = tokenizer.s2i['<sos>']
        self.eos_value = tokenizer.s2i['<eos>']
        self.pad_value = tokenizer.s2i['<pad>']
        self.max_len = 128
    
    # def forward(self, protein_inputs, drug_inputs):
        
    def forward(self, protein_embeddings, cruppted_inputs, input_mask, targets):
        
        print(protein_embeddings.shape)
        
        protein_inputs = self.pos_encoder(protein_embeddings)
        print(protein_inputs.shape)
        encoded_proteins = self.encoder(protein_inputs)
        # quantized, vq_loss = self.vq_layer(encoded_proteins)
        
        # print(encoded_proteins.)
        
        print(encoded_proteins.shape)
        
        # fake_mask
        mem_padding_mask = protein_inputs == -100
        print(mem_padding_mask)
        
        _, target_length = targets.shape
        target_mask = torch.triu(torch.ones(target_length, target_length, dtype=torch.bool),
                                 diagonal=1).to(targets.device)
        target_embed = self.word_embed(targets)
        target_embed = self.pos_encoding(target_embed.permute(1, 0, 2).contiguous())

        # predict
        output = self.decoder(target_embed, encoded_proteins,
                              x_mask = target_mask, mem_padding_mask = mem_padding_mask).permute(1, 0, 2).contiguous()
        prediction_scores = self.word_pred(output)
        
        # lm_loss
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        lm_loss = F.cross_entropy(shifted_prediction_scores.view(-1, self.vocab_size), targets.view(-1),
                                  ignore_index=self.pad_value)
        
        return prediction_scores, lm_loss
        

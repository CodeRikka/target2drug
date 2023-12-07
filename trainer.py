import torch
from torch.utils.data import DataLoader

from vqvae_test import VQVAE
from tokenizer import Tokenizer
import numpy as np

from dataset_test import SemiSmilesDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


from tqdm.auto import tqdm


from torch.optim import AdamW

file_path = 'data/processed_data_clean.csv'

dataset = SemiSmilesDataset(file_path=file_path)

input_dim = 768
hidden_dim = 1024
num_embeddings = 1024
embedding_dim = 768
num_heads = 8
num_layers = 8

from simple_ed import ED

# model = VQVAE(dataset.tokenizer, input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers).to('cuda:0')

model = ED(dataset.tokenizer, input_dim, hidden_dim, num_embeddings, embedding_dim, num_heads, num_layers).to('cuda:0')


loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True, collate_fn=dataset.collate_fn)


optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6, amsgrad=False)
scheduler = CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-6, last_epoch=-1)

for step, batch_data in tqdm(enumerate(loader), total=len(loader)):

    protein_emmbeddings, corrupted_inputs, input_mask, target_seqs = [i.to('cuda:0') for i in batch_data]
        
    # print(protein_emmbedding)
    # print(corrupted_inputs)
    # print(input_mask)
    # print(target_seqs)
        
        # print(protein_emmbedding, corrupted_inputs, input_mask, target_seqs)
    batch_size = corrupted_inputs.shape[0]
    
    prediction_scores, lm_loss, vq_loss = model(protein_emmbeddings, corrupted_inputs, input_mask, target_seqs)
    
    # NG
    beta = 0.2
    loss = lm_loss + vq_loss * beta
    
    
    loss.backward()
    # model(protein_emmbedding, corrupted_inputs, input_mask, target_seqs)
    
    
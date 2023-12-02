import torch
from torch.utils.data import DataLoader

from vqvae_test import VQVAE
from tokenizer import Tokenizer
import numpy as np

from dataset_test import SemiSmilesDataset

file_path = 'data/processed_data_clean.csv'

dataset = SemiSmilesDataset(file_path=file_path)

loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
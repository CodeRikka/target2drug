from tokenizer import Tokenizer
from read_data import read_data

import numpy as np
import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from typing import List

def _corrupt(token_seq: List[int], mask_token, corrupt_percent=0.1, poisson_lambda=2):
    # infilling, not perfect
    token_seq = token_seq.copy()
    l = len(token_seq)
    n = int(l * corrupt_percent)

    c = 0
    idx = sorted(np.random.choice(list(range(1, l - 1)), n), reverse=True)  # skip <sos>
    for i in idx:
        li = np.random.poisson(poisson_lambda)
        while li < 1:
            li = np.random.poisson(poisson_lambda)
        token_seq[i] = mask_token
        li -= 1
        p = i + 1
        while p < l and li > 0:
            del token_seq[p]
            l -= 1
            li -= 1
            c += 1
        if c >= n:
            break

    return token_seq

class SemiSmilesDataset(Dataset):

    # def __init__(self, file_path, tokenizer: Tokenizer,
    #              use_random_input_smiles=False, use_random_target_smiles=False, rsmiles=None, corrupt=True):
    def __init__(self, file_path, 
                 use_random_input_smiles=False, use_random_target_smiles=False, rsmiles=None, corrupt=True):
        """
        :param smiles_list: list of valid smiles
        :param tokenizer:
        :param use_random_input_smiles:
        :param use_random_target_smiles:
        :param rsmiles:
        :param corrupt: boolean, whether to use infilling scheme to corrupt input smiles
        """
        super().__init__()
        
        self.smiles_list, self.protein_list = read_data(file_path)
        
        tokenizer = Tokenizer(Tokenizer.gen_vocabs(set(self.smiles_list)))
        # print(tokenizer)
        
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.SPECIAL_TOKENS.index('<mask>')

        self.vocab_size = len(tokenizer)
        self.len = len(self.smiles_list)
        
        self.use_random_input_smiles = use_random_input_smiles
        self.use_random_target_smiles = use_random_target_smiles
        self.rsmiles = rsmiles
        self.corrupt = corrupt
        
        if rsmiles is None and (use_random_input_smiles or use_random_target_smiles):
            print('WARNING: The result of rdkit.Chem.MolToSmiles(..., doRandom=True) is NOT reproducible '
                  'because this function does not provide a way to control its random seed.')

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        protein_embedding = self.protein_list[item]
        # protein_embedding = torch.Tensor(protein_embedding)
        
        smiles = self.smiles_list[item]
        mol = Chem.MolFromSmiles(smiles)
        
        # clear isotope
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        
        csmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, doRandom=False)
        if self.rsmiles is not None:
            rsmiles = self.rsmiles[item]
        else:
            rsmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)
        
        input_smiles = rsmiles if self.use_random_input_smiles else csmiles
        target_smiles = rsmiles if self.use_random_target_smiles else csmiles
        
        input_seq = self.tokenizer.parse(input_smiles)
        target_seq, atom_idx = self.tokenizer.parse(target_smiles, return_atom_idx=True)
        
        if self.corrupt:
            corrupted_input = _corrupt(input_seq, self.mask_token)
        else:
            corrupted_input = input_seq
        
        corrupted_input = torch.LongTensor(corrupted_input)
        
        target_seq = torch.LongTensor(target_seq)

        # pp_graph, mapping = smiles2ppgraph(target_smiles)
        # pp_graph.ndata['h'] = \
        #     torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
        # pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
        
        # mapping = torch.FloatTensor(mapping)
        # mapping[:,pp_graph.num_nodes():] = -100  # torch cross entropy loss ignores -100 by default
        
        # mapping_ = torch.ones(target_seq.shape[0], MAX_NUM_PP_GRAPHS)*-100
        # mapping_[atom_idx,:] = mapping
        
        # return corrupted_input, pp_graph, mapping_, target_seq
        return protein_embedding, corrupted_input, target_seq

    @staticmethod
    def collate_fn(batch):
        pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>')

        # corrupted_inputs, pp_graphs, mappings, target_seqs, *other_descriptors = list(zip(*batch))
        protein_emmbedding, corrupted_inputs, target_seqs, *other_descriptors = list(zip(*batch))

        protein_emmbeddings = \
            pad_sequence(protein_emmbedding, batch_first=True, padding_value=-100)
        
        corrupted_inputs = \
            pad_sequence(corrupted_inputs, batch_first=True, padding_value=pad_token)
        input_mask = (corrupted_inputs==pad_token).bool()
        
        # pp_graphs = dgl.batch(pp_graphs)
        
        # mappings = pad_sequence(mappings, batch_first=True, padding_value=-100)  # torch cross entropy loss ignores -100 by default, but we do not use cross_entropy_loss acctually
        
        target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=pad_token)

        # return corrupted_inputs, input_mask, pp_graphs, mappings, target_seqs
        return protein_emmbeddings, corrupted_inputs, input_mask, target_seqs
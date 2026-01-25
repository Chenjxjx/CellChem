import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import anndata as ad
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
from dataset.tokenizer import MolTranBertTokenizer
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
import hdf5plugin
import anndata
from scipy.stats import pearsonr

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]
mol = pd.read_csv('data/cmap_drugbank_mapping.csv')
#from anndata import AnnData
adata = anndata.read_h5ad('./data/output_data_finetune.h5ad')
adata_cmap = adata
adata = anndata.read_h5ad('./data/ouput_xpr.h5ad')
adata_xpr = adata




def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1
    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits
    right_difits = np.digitize(x, bins, right=True)
    rands = np.random.rand(len(x))  # uniform random numbers
    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits

def read(data_path):
    CPI = pd.read_csv(data_path)
    drug = CPI['mol'].tolist() 
    smiles = CPI['smiles'].tolist() 
    protein = CPI['protein'].tolist()        
    label = CPI['label'].tolist()   
            
    
    return drug, smiles, protein, label


def read_seq_embedding(protein,data_path_1):
    Protein = np.load(data_path_1 + '/'+ protein + '.npy')
    Protein_len = Protein.shape[1]  
    len = 1024-Protein.shape[1]
    Protein = torch.tensor(Protein[0:,])
    Protein =torch.nn.functional.pad(Protein,pad=(0,0,0,len), mode='constant', value=1)
    Protein = Protein[0,:,:]
    return  Protein, Protein_len

        

def read_CRISPR_Cmap_label(Mol, Protein, label):
    adata = adata_xpr[adata_xpr.obs['Entry'] == Protein,:]
    X = torch.tensor(adata.X)
    mean_x = torch.mean(X,axis = 0)
    max_val = torch.max(mean_x)
    min_val = torch.min(mean_x)
    n_bins = 51
    bins = np.linspace(min_val, max_val, n_bins)
    digits = _digitize(mean_x, bins)
    # no zero digits!
    assert digits.min() >= 1
    assert digits.max() <= n_bins
    CRISPR = torch.tensor(digits)
    pert_id = list(set(mol.loc[mol['drug']== Mol]['pert_id'].tolist()))
    #print(Mol,pert_id)
    adatas = adata_cmap[adata_cmap.obs['pert_id'] == pert_id[0],:]
    if len(pert_id)>1:
        for i in range(1,len(pert_id)):
            adata_1 = adata_cmap[adata_cmap.obs['pert_id'] == pert_id[i],:]
            adatas=[adatas,adata_1]
            adatas = ad.concat(adatas, merge = "same")
    X = torch.tensor(adatas.X)
    mean_x = torch.mean(X,axis = 0)
    max_val = torch.max(mean_x)
    min_val = torch.min(mean_x)
    n_bins = 51
    bins = np.linspace(min_val, max_val, n_bins)
    digits = _digitize(mean_x, bins)
    # no zero digits!
    assert digits.min() >= 1
    assert digits.max() <= n_bins
    Cmap = torch.tensor(digits)
    return Cmap,CRISPR,label

class MoleculeDataset(Dataset):
    def __init__(self, data_path, data_path_1):
        super(Dataset, self).__init__()
        self.drug, self.smiles, self.protein, self.label= read(data_path)
        self.data_path_1 = data_path_1
    def __getitem__(self, index):
        Cmap, CRISPR, label= read_CRISPR_Cmap_label(self.drug[index], self.protein[index],self.label[index])
        Mol = self.smiles[index]
        mol = Chem.MolFromSmiles(self.smiles[index])
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()
        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        Mol = Data(x, edge_index=edge_index, edge_attr=edge_attr)
        Mol_len = len(Mol)
        label = int(self.label[index])
        Protein, Protein_len = read_seq_embedding(self.protein[index], self.data_path_1)
        label = int(label)
        idx = torch.LongTensor([index])
        return CRISPR,Cmap,Protein,Mol,label,Protein_len, Mol_len,idx

    def __len__(self):
        return len(self.drug)
    
    def len(self) -> int:
        return super().len()
    def get(self, idx: int) :
        return super().get(idx)
    
class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, data_pt,data_pv, data_path_1):
        super(object, self).__init__()
        self.data_pt = data_pt
        self.data_pv = data_pv
        self.data_path_1 = data_path_1
        self.batch_size = batch_size
        self.num_workers = 0

    def get_dataset(self):
        train_dataset = MoleculeDataset(data_path=self.data_pt, data_path_1=self.data_path_1)
        valid_dataset= MoleculeDataset(data_path=self.data_pv, data_path_1=self.data_path_1)
        return train_dataset,valid_dataset
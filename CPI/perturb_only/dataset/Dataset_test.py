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
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import anndata as ad
from torch_scatter import scatter
from torch.utils.data import  Dataset, DataLoader
from dataset.tokenizer import MolTranBertTokenizer
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
import hdf5plugin
import anndata
from scipy.stats import pearsonr

mol = pd.read_csv('/home/test_zd/CPI/cmap_drugbank_mapping_1.csv')
#from anndata import AnnData
adata = anndata.read_h5ad('/home/test_zd/scGPT/Model_fusion/examples/output_data/output_data_finetune.h5ad')
adata_cmap = adata


adata = anndata.read_h5ad('/home/test_zd/scGPT/Model_fusion/examples/output_data/ouput_xpr.h5ad')
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
    #sig_id_mol = CPI['sig_id_mol'].tolist()    
    #sig_id_protein = CPI['sig_id_protein'].tolist()
    drug = CPI['mol'].tolist()   
    protein = CPI['protein'].tolist()        
    label = CPI['label'].tolist()   
            
    
    return drug, protein, label


def read_seq_embedding(drug, protein, data_path_3,data_path_4):
    Mol = np.load(data_path_3 + '/'+ drug + '.npy')
    Protein = np.load(data_path_4 + '/'+ protein + '.npy')
    len = 1024-Protein.shape[1]
    Protein = torch.tensor(Protein[0:,])
    Protein =torch.nn.functional.pad(Protein,pad=(0,0,0,len), mode='constant', value=1)
    Protein = Protein[0,:,:]
    #print(Protein.shape)
    Mol = torch.tensor(Mol)
    len_mol = 873-Mol.shape[0]
    #print(Mol.shape)
    Mol = torch.nn.functional.pad(Mol,pad=(0,0,0,len_mol), mode='constant', value=1)
    #print(Mol.shape)
    return Mol, Protein

        

def read_CRISPR_Cmap_label(data_path_1,data_path_2, Mol, Protein, label):
    
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
    CRISPR = digits



    

    
    pert_id = Mol
    #print(Mol,pert_id)
    adatas = adata_cmap[adata_cmap.obs['pert_id'] == pert_id,:]

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
    Cmap = digits


 
   
    return Cmap,CRISPR,label

class MoleculeDataset(Dataset):
    def __init__(self, data_path, data_path_1, data_path_2, data_path_3,data_path_4):
        super(Dataset, self).__init__()
        self.drug, self.protein, self.label= read(data_path)
        self.data_path_1 = data_path_1
        self.data_path_2 = data_path_2
        self.data_path_3 = data_path_3
        self.data_path_4 = data_path_4
    def __getitem__(self, index):

  
        Cmap, CRISPR, label= read_CRISPR_Cmap_label(self.data_path_1,self.data_path_2,self.drug[index], self.protein[index],self.label[index])
        #Mol, Protein = read_seq_embedding(self.drug[index], self.protein[index], self.data_path_3,self.data_path_4)
        Mol = self.drug[index]
        Protein = [0]
        label = int(label)
        
        return CRISPR,Cmap,Protein,Mol,label

    def __len__(self):
        return len(self.drug)
    
class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, data_path, data_path_1, data_path_2,data_path_3, data_path_4):
        super(object, self).__init__()
        self.data_path = data_path
        self.data_path_1 = data_path_1
        self.data_path_2 = data_path_2
        self.data_path_3 = data_path_3
        self.data_path_4 = data_path_4
        self.batch_size = batch_size
        self.num_workers = 0

    def get_dataset(self):
        dataset = MoleculeDataset(data_path=self.data_path,data_path_1=self.data_path_1,data_path_2=self.data_path_2,data_path_3=self.data_path_3,data_path_4=self.data_path_4)

        return dataset
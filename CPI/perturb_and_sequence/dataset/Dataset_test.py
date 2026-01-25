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
from torch_geometric.data import Data, Dataset, DataLoader
#from torch_scatter import scatter
#from torch.utils.data import  Dataset, DataLoader
from dataset.tokenizer import MolTranBertTokenizer
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem

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

def read(data_path):
    CPI = pd.read_csv(data_path)
    CPI = CPI.sample(frac = 1)
    drug = CPI['smiles'].tolist()   
    protein = CPI['protein'].tolist()        
    label = CPI['label'].tolist()   
    name = CPI['mol'].tolist()          
    
    return drug, protein, label,name


def read_seq_embedding(protein,data_path_4):
    
    Protein = np.load(data_path_4 + '/'+ protein + '.npy')
    Protein_len = Protein.shape[1]
    
    len = 1024-Protein.shape[1]
    Protein = torch.tensor(Protein[0:,])
    Protein =torch.nn.functional.pad(Protein,pad=(0,0,0,len), mode='constant', value=1)
    Protein = Protein[0,:,:]
    #print(Protein.shape)
 
    return  Protein, Protein_len



class MoleculeDataset(Dataset):
    def __init__(self, data_path, data_path_3,data_path_4):
        super(Dataset, self).__init__()
        self.drug, self.protein, self.label,self.name= read(data_path)
        self.data_path_3 = data_path_3
        self.data_path_4 = data_path_4
    def __getitem__(self, index):

        Mol = self.drug[index]
        mol = Chem.MolFromSmiles(self.drug[index])
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        # aromatic = []
        # sp, sp2, sp3, sp3d = [], [], [], []
        # num_hs = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())
            # aromatic.append(1 if atom.GetIsAromatic() else 0)
            # hybridization = atom.GetHybridization()
            # sp.append(1 if hybridization == HybridizationType.SP else 0)
            # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)

        # z = torch.tensor(atomic_number, dtype=torch.long)
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)
        # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
        #                     dtype=torch.float).t().contiguous()
        # x = torch.cat([x1.to(torch.float), x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
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
        
        Protein, Protein_len = read_seq_embedding(self.protein[index], self.data_path_4)
        

        
        return Protein,Mol,label,Protein_len, Mol_len,self.drug[index],self.name[index]

    def __len__(self):
        return len(self.drug)
    
    def len(self) -> int:
        return super().len()
    def get(self, idx: int) :
        return super().get(idx)
    
class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, data_path, data_path_3, data_path_4):
        super(object, self).__init__()
        self.data_path = data_path
        self.data_path_3 = data_path_3
        self.data_path_4 = data_path_4
        self.batch_size = batch_size
        self.num_workers = 0
    def get_dataset(self):
        dataset = MoleculeDataset(data_path=self.data_path,data_path_3=self.data_path_3,data_path_4=self.data_path_4)

        return dataset
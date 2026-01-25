"""Write the MGPPI dataset finetune dataset scripts"""
import os
import csv
import math
import time
import random
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  

from pandarallel import pandarallel
pandarallel.initialize(nb_workers = 20, progress_bar = True)

class MGPPIData(Data):

    def __init__(self, *args, **kwargs):
        super(MGPPIData, self).__init__(*args, **kwargs)
    
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'chem_desc':
            return None
        return super(MGPPIData, self).__cat_dim__(key, value, *args, **kwargs)
        
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'chem_desc':
            return 0
        return super(MGPPIData, self).__inc__(key, value, *args, **kwargs)
    
    
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    # print("About to generate scaffolds")
    # for ind, smiles in enumerate(dataset.smiles_data):
    #     if ind % log_every_n == 0:
    #         print("Generating scaffold %d/%d" % (ind, data_len))
    #     scaffold = _generate_scaffold(smiles)
    #     if scaffold not in scaffolds:
    #         scaffolds[scaffold] = [ind]
    #     else:
    #         scaffolds[scaffold].append(ind)
    pdtask = pd.DataFrame({'smiles':dataset.smiles_data})
    scaffold_list = pdtask.parallel_apply(lambda row: _generate_scaffold(row['smiles']), axis=1).tolist()
    for ind, scaffold in enumerate(scaffold_list):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

def balance_generate_scaffolds(smiles_data, log_every_n=1000000):
    scaffolds = {}
    data_len = len(smiles_data)
    print(data_len)

    # print("About to generate scaffolds")
    # for ind, smiles in enumerate(smiles_data):
    #     if ind % log_every_n == 0:
    #         print("Generating scaffold %d/%d" % (ind, data_len))
    #     scaffold = _generate_scaffold(smiles)
    #     if scaffold not in scaffolds:
    #         scaffolds[scaffold] = [ind]
    #     else:
    #         scaffolds[scaffold].append(ind)
    pdtask = pd.DataFrame({'smiles':smiles_data})
    scaffold_list = pdtask.parallel_apply(lambda row: _generate_scaffold(row['smiles']), axis=1).tolist()
    for ind, scaffold in enumerate(scaffold_list):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

def balance_scaffold_split(
    dataset, 
    valid_size, 
    test_size, 
    mix_mg = False,
):
    """
    This function is used to 0, 1, 2 PPIMG classification.
    We keep all the MG in externel dataset and Enamine HTS 
        and PPI as training set and validation set, test set.
    """
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []
    cluster = dataset.cluster
    mg_count = 0
    if mix_mg:
        test_inds = cluster[2]['index']
        mg_count = len(test_inds)
        cluster.pop(2)
    
    for k, v in cluster.items():
        train_size = 1.0 - valid_size - test_size
        scaffold_sets = balance_generate_scaffolds(v['smi'])

        train_cutoff = train_size * len(v['smi'])
        valid_cutoff = (train_size + valid_size) * len(v['smi'])
        train_inds_count = 0
        valid_inds_count = 0
        print("About to sort in scaffold sets")
        for scaffold_set in scaffold_sets:
            scaffold_set_len = len(scaffold_set)
            scaffold_set = [v['index'][ind] for ind in scaffold_set]
            if train_inds_count + scaffold_set_len > train_cutoff:
                if train_inds_count + valid_inds_count + scaffold_set_len > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
                    valid_inds_count += scaffold_set_len
            else:
                train_inds += scaffold_set
                train_inds_count += scaffold_set_len
    total = len(train_inds) + len(valid_inds) + len(test_inds) - mg_count
    print(f'##### train index number: {len(train_inds)}')
    print(f'##### validation index number: {len(valid_inds)}')
    print(f'##### test index number: {len(test_inds)}')
    print(f'##### MG mix on/off: {mix_mg} with MG count: {mg_count}, so `test index - MG number` should: {len(test_inds) - mg_count}')
    print(f'##### real ratio: {len(train_inds) / total}/{len(valid_inds) / total}/{(len(test_inds) - mg_count) / total}')
    print(f'##### Check whether meet the ratio: {1.0 - valid_size - test_size}/{valid_size}/{test_size}......')
    return train_inds, valid_inds, test_inds
    

def read_smiles(data_path, target):
    smiles_data, labels, tags, molnames, databases = [], [], [], [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    # binary classfication task while data has multi-class
                    # such as 0 (decoy), 1 (PPI), 2 (MG)
                    label = int(label)
                    tags.append(label)
                    molnames.append(row['name'])
                    databases.append(row['database'])
                    if label > 1:
                        label = 1
                    labels.append(label)
    print(len(smiles_data))
    return smiles_data, labels, tags, molnames, databases

def read_descriptors(data_path, descriptors: Optional[List[str]] = None):
    # first parse the all descriptors from the external txt file
    if descriptors is None:
        desc_file = os.path.join(os.path.dirname(data_path), 'descriptors.txt')
        assert os.path.exists(desc_file), f"if `None` descriptors used, in default use all the descriptors prepared in {desc_file}"
        with open(desc_file, 'r') as f:
            descriptors = f.readlines()
    pf = pd.read_csv(data_path)
    return np.stack([pf[_desc.strip()].apply(float).to_numpy() for _desc in descriptors], axis = 0)


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, descriptors:Union[List[str], str] = None):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels, self.tags, self.molnames, self.databases = read_smiles(data_path, target)
        if not isinstance(descriptors, str):
            self.descroptors = read_descriptors(data_path, descriptors)
        else:
             self.descroptors = None
        self._tag_cluster()
        
    def _tag_cluster(self):
        from collections import defaultdict
        self.cluster = defaultdict(dict)
        for idx, (smi, tag) in enumerate(zip(self.smiles_data, self.tags)):
            if tag not in self.cluster:
                self.cluster[tag]['smi'] = []
                self.cluster[tag]['index'] = []
                
            self.cluster[tag]['smi'].append(smi)
            self.cluster[tag]['index'].append(idx)
        
    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            cctag = atom.GetChiralTag()
            chirality_idx.append(CHIRALITY_LIST.index(cctag) if cctag in CHIRALITY_LIST else 0)
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bdir = bond.GetBondDir()
            bdirid = BONDDIR_LIST.index(bdir) if bdir in BONDDIR_LIST else 0
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                bdirid
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                bdirid
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        if self.descroptors is None:
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, moltag=str(self.tags[index]),
                        molname=self.molnames[index], database=self.databases[index])
        else:
            descroptors = self.descroptors[index]
            descroptors = torch.from_numpy(descroptors).float32()
            data = MGPPIData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, 
                             chem_desc=descroptors, moltag=str(self.tags[index]),
                             molname=self.molnames[index], database=self.databases[index])
        return data
    
    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, descriptors = None, task = 'classification', 
        splitting = 'balance_scaffold', binary_cls = True, mix_mg = False,
        debug_inds_output = './MG_inds_debug.csv'
    ):
        super(object, self).__init__()
        assert task == 'classification', 'task must be either regression or classification'
        assert splitting in ['random', 'scaffold', 'balance_scaffold']
        self.data_path = data_path
        self.batch_size = batch_size
        self.descriptors = descriptors
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        self.mix_mg = mix_mg
        self.debug_inds_output = debug_inds_output
    
    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, descriptors= self.descriptors)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader
    
    def MG_debug(
        self,
        train_dataset,
        train_idx, 
        valid_idx, 
        test_idx,
    ):
        smiles_data = []
        tags = []
        molnames = []
        databases = []
        split = []
        split_keys = ['train', 'val', 'test']
        print('Saving train, val, test indices for debug....')
        for i, idxs in enumerate([train_idx, valid_idx, test_idx]):
            smiles_data.extend([train_dataset.smiles_data[idx] for idx in idxs])
            tags.extend([train_dataset.tags[idx] for idx in idxs])
            molnames.extend([train_dataset.molnames[idx] for idx in idxs])
            databases.extend([train_dataset.databases[idx] for idx in idxs])
            split.extend([split_keys[i]] * len(idxs))
        pf = pd.DataFrame(
            {
                'smiles': smiles_data,
                'label': tags,
                'name': molnames,
                'database': databases,
                'split': split,
            }
        )
        data_num = pf.shape[0]
        dup_num = pf.drop_duplicates(subset=['smiles', 'label', 'name', 'database'], ignore_index = True).shape[0]
        print(
            f'Debug report: before dropping duplicates, the number of datapoints is {data_num}; after dropping duplicates is {dup_num}'
        )
        pf.to_csv(self.debug_inds_output, index = False)
        
    
    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)
        elif self.splitting == 'balance_scaffold':
            train_idx, valid_idx, test_idx = balance_scaffold_split(
                train_dataset, self.valid_size, self.test_size, mix_mg = self.mix_mg)
        
        if isinstance(self.debug_inds_output, str):
            self.MG_debug(train_dataset, train_idx, valid_idx, test_idx)
        
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        
        return train_loader, valid_loader, test_loader

import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import yaml
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
from MOLNCMAP import *
from easydict import EasyDict as ed
from rdkit import Chem
from Model_mol.graphtransformer_molclr import GraphTransformer
import anndata as ad
import torch
import anndata
from anndata import AnnData
import scanpy as sc
import hdf5plugin
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
# from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from scgpt.tokenizer.gene_tokenizer import GeneVocab

sys.path.append("../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

from sm_perturb_ft import (
    parse_cp_gctx,
    preprocessing,
    clue_binning,
    sm_tokenize_and_pad_batch,
    molgraph_tokenize,
    train_test_split_by_scaffold,
)
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="clue",
    do_train=True,
    load_model="save/dev_clue-May20-00-03",
    load_mol_model="Model_mol/May08_16-21-26/checkpoints",
    adata_path="/lustre3/lhlai_pkuhpc/zhujt/projects/smilesncmap/data/clue_cp_level5_prepared.h5ad",
    mask_ratio=0.,
    epochs=100,
    n_bins=51,
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    lr=5e-4,
    batch_size=256,#64,
    layer_size=128,
    nlayers=4,
    nhead=4,
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT_256_300",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
if config.load_model is None:
    raise ValueError('no load model in config!')
print(config)

set_seed(config.seed)

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 968  # number of landmark gene
max_seq_len = n_hvg + 1
per_seq_batch_sample = False
DSBN = True  # Domain-spec batchnorm
explicit_zero_prob = True  # whether explicit bernoulli for zeros

dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
# save the whole script to the dir
os.system(f"cp {__file__} {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


# ## Loading and preparing data
if not Path(config.adata_path).exists():
    logger.info('Pre-processing the clue gctx file to AnnData...')
    adata = parse_cp_gctx(to_adata=True)
    adata = preprocessing(adata)
    adata.write_h5ad(
        config.adata_path,
        compression=hdf5plugin.FILTERS["zstd"],
        compression_opts=hdf5plugin.Zstd(clevel=5).filter_options
    )
else:
    logger.info('Read pre-prepared h5ad file...')
    adata = anndata.read_h5ad(config.adata_path) # 638737 × 968

    
ori_batch_col = 'bead_batch'
adata.obs["celltype"] = adata.obs["cell_iname"].astype("category")
adata.var = adata.var.reset_index().set_index('gene_symbols')
adata.var.rename({'rid': 'gene_id'}, axis = 1, inplace = True)
data_is_raw = False

# make the batch category column
adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

model_dir = Path(config.load_model)
model_config_file = model_dir / "args.json"
model_file = model_dir / "model_e14.pt"
vocab_file = model_dir / "vocab.json"

cond_tokens = adata.obs['str_cond'].astype('category').cat.categories.tolist()
if (model_dir / "vocab_all.json").exists():
    vocab = GeneVocab.from_file(model_dir / "vocab_all.json")
else:
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens + cond_tokens:
        if s not in vocab:
            vocab.append_token(s)
vocab.save_json(save_dir / "vocab_all.json")

# filter invalid smiles
def check_invalid_smiles(smi):
    smi = str(smi)
    if smi in ['nan', 'restricted']:
        return False

    mol = Chem.MolFromSmiles(smi)
    return mol is not None

valid_smiles_flag = adata.obs["canonical_smiles"].apply(check_invalid_smiles)
adata = adata[valid_smiles_flag, :]


adata.var["id_in_vocab"] = [
    1 if gene in vocab else -1 for gene in adata.var["gene_name"]
]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
logger.info(
    f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
    f"in vocabulary of size {len(vocab)}."
)
adata = adata[:, adata.var["id_in_vocab"] >= 0]
max_seq_len = adata.n_vars + 1
logger.info(f'Max sequence length: {max_seq_len}')

# model
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
logger.info(
    f"Resume model from {model_file}, the model args will be overriden by the "
    f"config {model_config_file}."
)
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

# set up the preprocessor, use the args to config the workflow
clue_binning(
    adata,
    key_to_process = 'X',
    result_binned_key = 'X_binned',
    n_bins = config.n_bins,
)



# ## Tokenize input
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

conds = adata.obs["str_cond"].tolist()
cond_ids = np.array(vocab(conds), dtype=int)

celltypes_labels = adata.obs["celltype"].tolist()
num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

smiles = adata.obs["canonical_smiles"].tolist()
num_compounds = len(set(smiles))
smiles = np.array(smiles)



train_data,train_celltype_labels,train_batch_labels,train_cond_labels, train_smiles= all_counts, celltypes_labels, batch_ids, cond_ids,smiles


tokenized_train = sm_tokenize_and_pad_batch(
    train_data,
    gene_ids,
    train_cond_labels,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)


def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )



    input_gene_ids_train = tokenized_train["genes"]

    
    input_values_train = masked_values_train
    target_values_train= tokenized_train["values"]
        
    

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()

    
    input_smiles_train = train_smiles

    
    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        #print(train_sort_ids.shape)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        input_smiles_train = input_smiles_train[train_sort_ids]



    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "smiles": input_smiles_train,
    }


    return train_data_pt

def _load_pre_trained_weights(model):
    try:
        model_dir = Path(config.load_mol_model)    
        state_dict = torch.load(model_dir/'model.pth', map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")

    return model
# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        smiles = data["smiles"]
        graph = molgraph_tokenize(smiles)
        data['graph'] = graph
        return data


# data_loader
def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

# # Create and finetune scGPT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model_scGPT = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=True,
    use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    n_input_bins=n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
)






config_file = f'{config.load_mol_model}/config_mg.yaml'
config_mol = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
model_scGPT.to(device)
graphtransformer = GraphTransformer(**config_mol["model"])

model =  Smilesncmap(model_scGPT,graphtransformer).to(device)

model.load_state_dict(torch.load(model_file))

print(model)

wandb.watch(model)


nt_xent_criterion = NTXentLoss(device, config['batch_size'], **config_mol['loss'])
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

train_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=config.batch_size,
    shuffle=False,
    intra_domain_shuffle=False,
    drop_last=False,
)
loader = train_loader
model.eval()
total_loss= 0.0

log_interval = config.log_interval
start_time = time.time()

MOL = torch.randn(1,512) 
CELL = torch.randn(1,512) 

num_batches = len(loader)
for batch, batch_data in enumerate(loader):
    input_gene_ids = batch_data["gene_ids"].to(device)
    input_values = batch_data["values"].to(device)
    target_values = batch_data["target_values"].to(device)
    batch_labels = batch_data["batch_labels"].to(device)
    graphs = batch_data["graph"].to(device)
    # smiles = batch_data["smiles"]
    src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
        cmap_embedding,mol_embedding = model(
            input_gene_ids,
            input_values,
            src_key_padding_mask,
            batch_labels,
            graphs,  
        )
        mol_embedding = F.normalize(mol_embedding, dim=1).cpu()
        cmap_embedding = F.normalize(cmap_embedding, dim=1).cpu()
        #print(cmap_embedding.shape,mol_embedding.shape)
        
        MOL = torch.cat((MOL, mol_embedding), 0)
        CELL = torch.cat((CELL, cmap_embedding), 0)

       
        
        #CELL= CELL.cpu()
MOL =  MOL[1:,:].numpy()
CELL = CELL[1:,:].numpy()
print(CELL.shape)
CELL = CELL / np.linalg.norm(CELL, axis=1, keepdims=True)
        
adata.obsm["X_scGPT"] = CELL
adata.obsm["Mol_emb"] = MOL 
adata.write_h5ad(
    save_dir / 'output_adata.h5ad',
    compression=hdf5plugin.FILTERS["zstd"],
    compression_opts=hdf5plugin.Zstd(clevel=5).filter_options
)


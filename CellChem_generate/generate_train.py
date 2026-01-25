import copy
import gc
import json
import os
import random
from pathlib import Path
import sys
import time
import yaml
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
from generate import *
from easydict import EasyDict as ed
from rdkit import Chem
from Model_mol.graphtransformer_molclr import GraphTransformer
import csv
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
sys.path.append("./")
from scgpt.tokenizer.gene_tokenizer import GeneVocab

sys.path.append("./")
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
    clue_binning_control,
    sm_tokenize_and_pad_batch,
    molgraph_tokenize,
    random_k_fold_split
)
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"

hyperparameter_defaults = dict(
    seed=2024,
    dataset_name="clue",
    do_train=True,
    load_model="../CellChem_pretrain/CellChem/save/dev_clue-May20-00-03/",
    load_mol_model="../CellChem_pretrain/mol_gt_pretrain/ckpt/May08_16-21-26/checkpoints",
    adata_path="data_generate/clue_cp_level5_prepared_with_cell_rep_generate_random_train.h5ad",  ###(scaffold,celltype)
    mask_ratio=0.0, 
    epochs=100,  
    n_bins=51,
    GEPC=True,
    ecs_thres=0.8,
    dab_weight=1.0,
    lr=5e-4,
    batch_size=64,   
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.0,
    schedule_ratio=0.97,
    save_eval_interval=2,  
    log_interval=100,
    fast_transformer=True,
    pre_norm=True,
    amp=True,
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

# ---- Fast reproducible seeding for DataLoader (no deterministic slowdown) ----
def seed_worker(worker_id: int):
    # Make numpy/python RNG in each worker deterministic across runs.
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

_DATALOADER_GEN = torch.Generator()
_DATALOADER_GEN.manual_seed(int(config.seed))
# -----------------------------------------------------------------------------


# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 968  # number of landmark gene
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
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
print(adata)
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
model_file = model_dir / "best_model.pt"
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
#print(adata.X.shape)
# set up the preprocessor, use the args to config the workflow
clue_binning(
    adata,
    key_to_process = 'X',
    result_binned_key = 'X_binned',
    n_bins = config.n_bins,
)
print(adata.obsm['cell_rep'].shape)
clue_binning_control(
    adata,
    key_to_process = 'cell_rep',
    result_binned_key = 'X_binned_cell',
    n_bins = config.n_bins,
)

if per_seq_batch_sample:
    # sort the adata by batch_id in advance
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

# ## Tokenize input
input_layer_key = "X_binned"
control_layer_key = "X_binned_cell"
control_counts =  (
    adata.layers[control_layer_key].A
    if issparse(adata.layers[control_layer_key])
    else adata.layers[control_layer_key]
)

all_counts = adata.X
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



k_folds = int(getattr(config, 'k', getattr(config, 'kfold', 5)))
folds = list(random_k_fold_split(all_counts, control_counts, celltypes_labels, batch_ids, cond_ids, smiles_data=smiles, shuffle=True, k=k_folds))
print(f'[KFold] prepared {len(folds)} folds (k={k_folds}).')
if len(folds) == 0:
    raise RuntimeError('No folds were generated. Check dataset size and k in random_k_fold_split.')



def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    
    input_smiles_train = train_smiles
    input_smiles_valid = valid_smiles

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]  ###control
        true_values_train = train_data[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        input_smiles_train = input_smiles_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        true_values_valid = valid_data[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        input_smiles_valid = input_smiles_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "true_label":true_values_train,   ####true
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "smiles": input_smiles_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "true_label":true_values_valid,   ####true
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "smiles": input_smiles_valid,
    }

    return train_data_pt, valid_data_pt


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
            worker_init_fn=seed_worker,
            generator=_DATALOADER_GEN,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=_DATALOADER_GEN,
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
    use_batch_labels=False,
    num_batch_labels=num_batch_types,
    n_input_bins=n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        # Ensure CPU-compatible deserialization when running without CUDA
        model_p = torch.load(model_file, map_location=torch.device('cpu'))
        mol_state = {}
        for param_tensor in model_p:
            if 'mol_encoder' in param_tensor:
                mol_state.update({param_tensor[12:]:model_p[param_tensor]})
        cmap_state = {}
        for param_tensor in model_p:
            if 'cmap_encoder' in param_tensor:
                cmap_state.update({param_tensor[13:]:model_p[param_tensor]})


        
        model_scGPT.load_state_dict(cmap_state)
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model_scGPT.state_dict()
        pretrained_dict = torch.load(model_file, map_location=torch.device('cpu'))
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model_scGPT.load_state_dict(model_dict)

model_dir = Path(config.load_mol_model)

config_file = f'{config.load_mol_model}/config_mg.yaml'
config_mol = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
model_scGPT.to(device)
graphtransformer = GraphTransformer(**config_mol["model"])
graphtransformer.load_state_dict(mol_state)
graphtransformer.to(device)
model = Generation(model_scGPT,graphtransformer).to(device)


wandb.watch(model)

backbone_params = list(model.scgpt.parameters())
head_params = list(model.mol_encoder.parameters()) + list(model.decoder.parameters())

# ---------------- Strict K-Fold support ----------------
# Snapshot the initial weights AFTER loading pretrained weights.
# Every fold will start from exactly this state (no leakage across folds).
init_model_state = copy.deepcopy(model.state_dict())

def _reset_optim_state():
    """Re-create optimizer/scheduler/scaler for each fold."""
    global optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1, gamma=config.schedule_ratio
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# Initialize once (will be re-initialized at each fold start)
_reset_optim_state()
# -------------------------------------------------------


def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    kl_div = -0.5 * torch.sum(1 + logvar_q - logvar_p - (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp(), dim=1)
    return kl_div.mean()

def batch_spearman_correlation_torch(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    
    def rank_data(x):
        batch_size, dim = x.shape
        ranks = torch.zeros_like(x)
        
        for i in range(batch_size):
            sorted_vals, indices = torch.sort(x[i])
            unique_vals, counts = torch.unique_consecutive(sorted_vals, return_counts=True)
            
            cum_counts = torch.cumsum(counts, dim=0)
            ranks_for_unique = 0.5 * (2 * cum_counts - counts + 1)
            
            expanded_ranks = torch.repeat_interleave(ranks_for_unique, counts)
            ranks[i] = expanded_ranks[torch.argsort(indices)]
            
        return ranks
    
    rank1 = rank_data(matrix1)
    rank2 = rank_data(matrix2)
    
    mean1 = rank1.mean(dim=1, keepdim=True)  # (batch, 1)
    mean2 = rank2.mean(dim=1, keepdim=True)  # (batch, 1)
    
    centered1 = rank1 - mean1
    centered2 = rank2 - mean2
    
    covariance = (centered1 * centered2).sum(dim=1)  # (batch,)
    std1 = torch.sqrt(torch.sum(centered1 ** 2, dim=1))  # (batch,)
    std2 = torch.sqrt(torch.sum(centered2 ** 2, dim=1))  # (batch,)
    
    eps = 1e-8
    correlations = covariance / (std1 * std2 + eps)  # (batch,)
    
    return correlations

def batch_pearson_correlation_torch(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    
    mean1 = matrix1.mean(dim=1, keepdim=True)  # (batch, 1)
    mean2 = matrix2.mean(dim=1, keepdim=True)  # (batch, 1)
    
    centered1 = matrix1 - mean1
    centered2 = matrix2 - mean2
    
    covariance = (centered1 * centered2).sum(dim=1)  # (batch,)
    std1 = torch.sqrt(torch.sum(centered1 ** 2, dim=1))  # (batch,)
    std2 = torch.sqrt(torch.sum(centered2 ** 2, dim=1))  # (batch,)
    
    eps = 1e-8
    correlations = covariance / (std1 * std2 + eps)  # (batch,)
    
    return correlations

import torch

def batch_precision_k(label_test, label_predict, k, num_pos=200, num_neg=200):
    test_ranks = torch.argsort(label_test, dim=1)      # [batch, num_classes]
    pred_ranks = torch.argsort(label_predict, dim=1)   # [batch, num_classes]
    
    true_neg = test_ranks[:, :num_neg]                # [batch, num_neg]
    true_pos = test_ranks[:, -num_pos:]               # [batch, num_pos]
    pred_neg = pred_ranks[:, :k]                      # [batch, k]
    pred_pos = pred_ranks[:, -k:]                     # [batch, k]
    
    def batch_intersection(a, b):
        # a: [batch, m], b: [batch, n] ->  [batch,]
        expand_a = a.unsqueeze(2).expand(-1, -1, b.size(1))  # [batch, m, n]
        expand_b = b.unsqueeze(1).expand(-1, a.size(1), -1)  # [batch, m, n]
        matches = (expand_a == expand_b).any(dim=2).sum(dim=1)  # [batch,]
        return matches.float()
    
    # Precision
    neg_intersect = batch_intersection(true_neg, pred_neg)  # [batch,]
    pos_intersect = batch_intersection(true_pos, pred_pos)  # [batch,]
    
    precision_neg = neg_intersect / k  #  Precision
    precision_pos = pos_intersect / k  #  Precision
    return precision_neg, precision_pos

import numpy as np
from sklearn.metrics import ndcg_score

def batch_ndcg(label_test, label_predict):
    sort_indices = torch.argsort(label_test, dim=1, descending=True)  

    sorted_true = torch.gather(label_test, 1, sort_indices)
    sorted_pred = torch.gather(label_predict, 1, sort_indices)

    batch_ndcg_scores = torch.zeros(label_test.size(0), device=label_test.device)
    
    for i in range(label_test.size(0)):
        ranks = torch.arange(2, label_test.size(1)+2, device=label_test.device)
        dcg = (sorted_true[i] / torch.log2(ranks)).sum()
        ideal_dcg = (torch.sort(sorted_true[i], descending=True)[0] / torch.log2(ranks)).sum()
        batch_ndcg_scores[i] = dcg / (ideal_dcg + 1e-8)  
    return batch_ndcg_scores

def masked_cosine_loss(pred: torch.Tensor,
                       label: torch.Tensor,
                       src_key_padding_mask: torch.Tensor,
                       eps: float = 1e-8,
                       center: bool = True):
    valid = (~src_key_padding_mask).float()[:,1:] 
    #print(valid)
    #print(valid.shape)

    pred = pred * valid
    label = label * valid

    if center:
        n = valid.sum(dim=1).clamp_min(2.0)
        pred_mean = pred.sum(dim=1, keepdim=True) / n.unsqueeze(1)
        lab_mean  = label.sum(dim=1, keepdim=True) / n.unsqueeze(1)
        pred = (pred - pred_mean) * valid
        label = (label - lab_mean) * valid


    pred_norm = torch.sqrt((pred * pred).sum(dim=1) + eps)
    lab_norm  = torch.sqrt((label * label).sum(dim=1) + eps)
    cos = (pred * label).sum(dim=1) / (pred_norm * lab_norm + eps)

    loss = (1.0 - cos).mean() / 0.5 
    return loss

def mixed_loss(pred, label, src_key_padding_mask,
               alpha_cos=1.0, alpha_huber=0.1, huber_beta=1.0):
    valid = (~src_key_padding_mask).float()[:,1:]
    # cosine
    cos_loss = masked_cosine_loss(pred, label, src_key_padding_mask, center=True)
    # huber (masked)
    huber = torch.nn.SmoothL1Loss(reduction="none", beta=huber_beta)
    hub = huber(pred, label)
    hub = (hub * valid).sum() / valid.sum().clamp_min(1.0)
    return alpha_cos * cos_loss + alpha_huber * hub

def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_num =0
    total_loss= 0.0
    log_interval = config.log_interval
    start_time = time.time()
    num_batches = len(loader)
    P = []
    S = []
    N10 = []
    N50 = []
    N100 = []
    P10 = []
    P50 = []
    P100 = []
    NDCG = []
    for batch, batch_data in enumerate(loader):

        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        true_values = batch_data["true_label"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        graphs = batch_data["graph"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            cmap_embedding,mol_embedding, cell_mol_tilde  = model(
                input_gene_ids,
                target_values,
                src_key_padding_mask,
                graphs,  
            )
            mol_embedding = F.normalize(mol_embedding, dim=1)
            cmap_embedding = F.normalize(cmap_embedding, dim=1)
            gene_means_1 = cell_mol_tilde[:,1:]
            #loss_func = torch.nn.MSELoss()
            loss = mixed_loss(gene_means_1, true_values, src_key_padding_mask,alpha_cos=1.0, alpha_huber=0.1, huber_beta=1.0)
            #loss = loss_func(gene_means_1, true_values)
            pearson = batch_pearson_correlation_torch(gene_means_1,true_values)
            spearman = batch_spearman_correlation_torch(gene_means_1,true_values)
            n10,p10 = batch_precision_k(true_values,gene_means_1,10)
            n50,p50 = batch_precision_k(true_values,gene_means_1,50)
            n100,p100 = batch_precision_k(true_values,gene_means_1,100)
            ndcg = batch_ndcg(true_values,gene_means_1)
            N10.append(n10)
            N50.append(n50)
            N100.append(n100)
            P10.append(p10)
            P50.append(p50)
            P100.append(p100)
            P.append(pearson)
            S.append(spearman)
            NDCG.append(ndcg)
            metrics_to_log = {"train/loss": loss.item()}
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "

                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_num += len(input_gene_ids)
    P = torch.cat(P, dim=0)
    S = torch.cat(S,dim=0)
    P10 = torch.cat(P10, dim=0)
    P50 = torch.cat(P50, dim=0)
    P100 = torch.cat(P100, dim=0)
    N10 = torch.cat(N10, dim=0)
    N50 = torch.cat(N50, dim=0)
    N100 = torch.cat(N100, dim=0)
    NDCG = torch.cat(NDCG, dim=0)
    PEARSON = torch.mean(P)
    SPEARMAN = torch.mean(S)
    PO10= torch.mean(P10)
    PO50= torch.mean(P50)
    PO100= torch.mean(P100)
    NEG10= torch.mean(N10)
    NEG50= torch.mean(N50)
    NEG100= torch.mean(N100)
    NDCG_mean= torch.mean(NDCG)
    return total_loss/total_num,PEARSON, SPEARMAN, NDCG_mean, PO10, PO50, PO100, NEG10, NEG50, NEG100




def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0

    P = []
    S = []
    N10 = []
    N50 = []
    N100 = []
    P10 = []
    P50 = []
    P100 = []
    NDCG = []
    with torch.no_grad():
        for batch_data in loader:
            true_values = batch_data["true_label"].to(device)
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            graphs = batch_data["graph"].to(device)

            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                cmap_embedding,mol_embedding, cell_mol_tilde  = model(
                    input_gene_ids,
                    target_values,
                    src_key_padding_mask,
                    graphs,  
                )
    
                mol_embedding = F.normalize(mol_embedding, dim=1)
                cmap_embedding = F.normalize(cmap_embedding, dim=1)
                gene_means_1 = cell_mol_tilde[:,1:]
                #loss_func = torch.nn.MSELoss()
                #loss = loss_func(gene_means_1, true_values)
                loss = mixed_loss(gene_means_1, true_values, src_key_padding_mask,alpha_cos=1.0, alpha_huber=0.1, huber_beta=1.0)
            
                pearson = batch_pearson_correlation_torch(gene_means_1,true_values)
                spearman = batch_spearman_correlation_torch(gene_means_1,true_values)
                n10,p10 = batch_precision_k(true_values,gene_means_1,10)
                n50,p50 = batch_precision_k(true_values,gene_means_1,50)
                n100,p100 = batch_precision_k(true_values,gene_means_1,100)
                ndcg = batch_ndcg(true_values,gene_means_1)
                N10.append(n10)
                N50.append(n50)
                N100.append(n100)
                P10.append(p10)
                P50.append(p50)
                P100.append(p100)
                P.append(pearson)
                S.append(spearman)
                NDCG.append(ndcg)

            total_loss += loss.item() * len(input_gene_ids)

            total_num += len(input_gene_ids)

        #print(PEARSON)
    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "epoch": epoch,
        },
    )

    P = torch.cat(P, dim=0) 
    S = torch.cat(S,dim=0)
    P10 = torch.cat(P10, dim=0) 
    P50 = torch.cat(P50, dim=0) 
    P100 = torch.cat(P100, dim=0) 
    N10 = torch.cat(N10, dim=0) 
    N50 = torch.cat(N50, dim=0) 
    N100 = torch.cat(N100, dim=0) 
    NDCG = torch.cat(NDCG, dim=0) 
    PEARSON = torch.mean(P)
    SPEARMAN = torch.mean(S)
    PO10= torch.mean(P10)
    PO50= torch.mean(P50)
    PO100= torch.mean(P100)
    NEG10= torch.mean(N10)
    NEG50= torch.mean(N50)
    NEG100= torch.mean(N100)
    NDCG_mean= torch.mean(NDCG)
    return total_loss / total_num,  PEARSON, SPEARMAN, NDCG_mean, PO10, PO50, PO100, NEG10, NEG50, NEG100



#
def extract_value(x):
    if hasattr(x, 'item'): 
        return x.item()
    return x

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()
results_csv = save_dir / "cross_validation_results.csv"
if not results_csv.exists():
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Fold', 'Best_Epoch', 
            'Train_Loss', 'Train_Pearson', 'Train_Spearman', 'Train_NDCG_Mean',
            'Train_PO10', 'Train_PO50', 'Train_PO100',
            'Train_NEG10', 'Train_NEG50', 'Train_NEG100',
            'Valid_Loss', 'Valid_Pearson', 'Valid_Spearman', 'Valid_NDCG_Mean',
            'Valid_PO10', 'Valid_PO50', 'Valid_PO100',
            'Valid_NEG10', 'Valid_NEG50', 'Valid_NEG100'
        ])


for i, (
    train_data,
    valid_data,
    train_control,
    valid_control,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
    train_cond_labels,
    valid_cond_labels,
    train_smiles,
    valid_smiles) in enumerate(folds):
    print(f"\nFold {i+1}:")
    print(f"  Training size: {len(train_data)}")
    print(f"  Validation size: {len(valid_data)}")

    # ---- STRICT K-FOLD RESET (no leakage across folds) ----
    model.load_state_dict(init_model_state, strict=True)
    _reset_optim_state()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    # ------------------------------------------------------
    train_data= torch.from_numpy(train_data).float()
    valid_data= torch.from_numpy(valid_data).float()

    tokenized_train = sm_tokenize_and_pad_batch(
        train_control,
        gene_ids,
        train_cond_labels,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    tokenized_valid = sm_tokenize_and_pad_batch(
        valid_control,
        gene_ids,
        valid_cond_labels,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=config.batch_size,
            shuffle=True, # False,
            intra_domain_shuffle=True,
            drop_last=False # False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False # False,
        )

        if config.do_train:
            train_loss, train_pearson, train_spearman, train_ndcg, train_po10, train_po50, train_po100, train_neg10, train_neg50, train_neg100 = train(
                model,
                loader=train_loader,
            )
        print(train_loss, train_pearson, train_spearman, train_ndcg,train_po10, train_po50, train_po100, train_neg10, train_neg50, train_neg100)
        valid_loss, valid_pearson, valid_spearman, valid_ndcg, valid_po10, valid_po50, valid_po100, valid_neg10, valid_neg50, valid_neg100 =evaluate(
            model,
            loader=valid_loader,
        )
        print(valid_loss, valid_pearson, valid_spearman, valid_ndcg, valid_po10, valid_po50, valid_po100, valid_neg10, valid_neg50, valid_neg100 )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {valid_loss:5.4f} | "
            f"pearson {valid_pearson:5.4f} | "
        )
        logger.info("-" * 89)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            best_metrics = {
            'Fold': i+1,
            'Best_Epoch': epoch,
            'Train_Loss': train_loss,
            'Train_Pearson': train_pearson,
            'Train_Spearman': train_spearman,
            'Train_NDCG_Mean': train_ndcg,
            'Train_PO10': train_po10,
            'Train_PO50': train_po50,
            'Train_PO100': train_po100,
            'Train_NEG10': train_neg10,
            'Train_NEG50': train_neg50,
            'Train_NEG100': train_neg100,
            'Valid_Loss': valid_loss,
            'Valid_Pearson': valid_pearson,
            'Valid_Spearman': valid_spearman,
            'Valid_NDCG_Mean': valid_ndcg,
            'Valid_PO10': valid_po10,
            'Valid_PO50': valid_po50,
            'Valid_PO100': valid_po100,
            'Valid_NEG10': valid_neg10,
            'Valid_NEG50': valid_neg50,
            'Valid_NEG100': valid_neg100
        }
            
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}_{i+1}fold.pt")


        scheduler.step()

    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
        best_metrics['Fold'],
        best_metrics['Best_Epoch'],
        extract_value(best_metrics['Train_Loss']),
        extract_value(best_metrics['Train_Pearson']),
        extract_value(best_metrics['Train_Spearman']),
        extract_value(best_metrics['Train_NDCG_Mean']),
        extract_value(best_metrics['Train_PO10']),
        extract_value(best_metrics['Train_PO50']),
        extract_value(best_metrics['Train_PO100']),
        extract_value(best_metrics['Train_NEG10']),
        extract_value(best_metrics['Train_NEG50']),
        extract_value(best_metrics['Train_NEG100']),
        extract_value(best_metrics['Valid_Loss']),
        extract_value(best_metrics['Valid_Pearson']),
        extract_value(best_metrics['Valid_Spearman']),
        extract_value(best_metrics['Valid_NDCG_Mean']),
        extract_value(best_metrics['Valid_PO10']),
        extract_value(best_metrics['Valid_PO50']),
        extract_value(best_metrics['Valid_PO100']),
        extract_value(best_metrics['Valid_NEG10']),
        extract_value(best_metrics['Valid_NEG50']),
        extract_value(best_metrics['Valid_NEG100'])
    ])
    # save the best model
    torch.save(best_model.state_dict(), save_dir / f"best_model_{i+1}fold.pt")
    # (Disabled) W&B artifact logging of large .pt files to avoid disk-quota issues.
    # Record the path instead (works for offline runs too).
    wandb.summary[f"best_model_path_fold{i+1}"] = str(save_dir / f"best_model_{i+1}fold.pt")

run.finish()
gc.collect()

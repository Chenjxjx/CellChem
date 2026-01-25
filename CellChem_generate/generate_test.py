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
from generate import *
import re
from easydict import EasyDict as ed
from rdkit import Chem
from Model_mol.graphtransformer_molclr import GraphTransformer
import csv
import argparse
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
    sm_tokenize_and_pad_batch,
    molgraph_tokenize,
    train_test_split_by_scaffold,
    clue_binning_control,
)
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="clue",
    do_train=False,
    load_model="./dev_clue-May20-00-03/",
    load_mol_model="./May08_16-21-26/checkpoints",
    adata_path="./data_generate/clue_cp_level5_prepared_with_cell_rep_generate_random_test.h5ad",  
    mask_ratio=0.,
    epochs=100,
    n_bins=51,
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    lr=5e-4,
    batch_size=64,#64,
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


parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, choices=["random", "celltype", "scaffold"], default=None,
                    help="Select the type of dataset split for evaluation:random/celltype/scaffold")
parser.add_argument("--model_dir", type=str, default="save/dev_clue-Jan03-05-12",
                    help="include best_model_{fold}fold.pt the training output directory")
args, unknown = parser.parse_known_args()

# Based on the scene switching data file and the output CSV file name
scenario_to_adata = {
    "random": "./data_generate/clue_cp_level5_prepared_with_cell_rep_generate_random_test.h5ad",
    "celltype": "./data_generate/clue_cp_level5_prepared_with_cell_rep_generate_celltype_test.h5ad",
    "scaffold": "./data_generate/clue_cp_level5_prepared_with_cell_rep_generate_scaffold_test.h5ad",
}

if args.scenario is not None:
    # The secure update method using wandb allows for the modification of existing key-value pairs.
    wandb.config.update({'adata_path': scenario_to_adata[args.scenario]}, allow_val_change=True)

result_output_dir = Path("./result_output")
result_output_dir.mkdir(parents=True, exist_ok=True)
if args.scenario is not None:
    out_csv_path = result_output_dir / f"model_evaluation_results_{args.scenario}.csv"
else:
    # Infer the scene based on the configured adata_path
    adata_path_str = str(config.adata_path).lower()
    inferred = "random"
    if "celltype" in adata_path_str:
        inferred = "celltype"
    elif "scaffold" in adata_path_str:
        inferred = "scaffold"
    out_csv_path = result_output_dir / f"model_evaluation_results_{inferred}.csv"
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
per_seq_batch_sample = True
DSBN = True  # Domain-spec batchnorm
explicit_zero_prob = True  # whether explicit bernoulli for zeros

dataset_name = config.dataset_name

# ## Loading and preparing data
if not Path(config.adata_path).exists():
#    logger.info('Pre-processing the clue gctx file to AnnData...')
    adata = parse_cp_gctx(to_adata=True)
    adata = preprocessing(adata)
    adata.write_h5ad(
        config.adata_path,
        compression=hdf5plugin.FILTERS["zstd"],
        compression_opts=hdf5plugin.Zstd(clevel=5).filter_options
    )
else:
#    logger.info('Read pre-prepared h5ad file...')
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
#vocab.save_json(save_dir / "vocab_all.json")

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


# model
with open(model_config_file, "r") as f:
    model_configs = json.load(f)

embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

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

(
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
    valid_smiles,
) = train_test_split_by_scaffold(
    all_counts, control_counts, celltypes_labels, batch_ids, cond_ids, smiles_data=smiles, test_size=0, shuffle=True
)

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

def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )


    input_gene_ids_train = tokenized_train["genes"]
    input_values_train= masked_values_train
    target_values_train = tokenized_train["values"]

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    
    input_smiles_train = train_smiles


    if sort_seq_batch:
         train_sort_ids = np.argsort(train_batch_labels)
         input_gene_ids_train = input_gene_ids_train[train_sort_ids]
         input_values_train = input_values_train[train_sort_ids]  ###control
         true_values_train = train_data[train_sort_ids]
         target_values_train = target_values_train[train_sort_ids]
         tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
         input_smiles_train = input_smiles_train[train_sort_ids]



    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "true_label":true_values_train,   ####true
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "smiles": input_smiles_train,
    }


    return train_data_pt


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
def load_matched_weights(model, checkpoint_path):
    model_state = model.state_dict()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    matched_weights = {
        k: v for k, v in checkpoint.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    model_state.update(matched_weights)
    model.load_state_dict(model_state)

    skipped = [k for k in checkpoint if k not in matched_weights]
    print("skip:", skipped)
    return model

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
        
        model_p = torch.load(model_file)
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
        pretrained_dict = torch.load(model_file)
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



optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    kl_div = -0.5 * torch.sum(1 + logvar_q - logvar_p - (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp(), dim=1)
    return kl_div.mean()

def batch_spearman_correlation_torch(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape,
    
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
    assert matrix1.shape == matrix2.shape,
    
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
    precision_pos = pos_intersect / k  # Precision
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
    total_loss= 0.0
    log_interval = config.log_interval
    start_time = time.time()
    num_batches = len(loader)
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
            loss = mixed_loss(gene_means_1, true_values, src_key_padding_mask,alpha_cos=1.0, alpha_huber=0.1, huber_beta=1.0)
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

    i=0
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
            i+=1
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

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            #"epoch": epoch,
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

def extract_tensor_value(tensor_str):
    match = re.search(r"[-+]?\d*\.\d+|\d+", tensor_str)
    return float(match.group()) if match else 0.0

train_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=config.batch_size,
    shuffle=True,
    intra_domain_shuffle=True,
    drop_last=False
)


with open(out_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'Model', 'LOSS_valid', 'PEARSON_valid', 'SPEARMAN_valid',
        'NDCG_mean_valid', 'PO10_valid', 'PO50_valid', 'PO100_valid',
        'NEG10_valid', 'NEG50_valid', 'NEG100_valid'
    ])

for fold in range(1, 6):
    model_path = os.path.join(args.model_dir, f"best_model_{fold}fold.pt")

    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found, skipping...")
        continue


    model = load_matched_weights(model, model_path)

    results = evaluate(
        model,
        loader=train_loader,
    )

    (LOSS_valid, PEARSON_valid, SPEARMAN_valid, NDCG_mean_valid,
     PO10_valid, PO50_valid, PO100_valid,
     NEG10_valid, NEG50_valid, NEG100_valid) = results

    PEARSON_valid = extract_tensor_value(str(PEARSON_valid))
    SPEARMAN_valid = extract_tensor_value(str(SPEARMAN_valid))
    NDCG_mean_valid = extract_tensor_value(str(NDCG_mean_valid))
    PO10_valid = extract_tensor_value(str(PO10_valid))
    PO50_valid = extract_tensor_value(str(PO50_valid))
    PO100_valid = extract_tensor_value(str(PO100_valid))
    NEG10_valid = extract_tensor_value(str(NEG10_valid))
    NEG50_valid = extract_tensor_value(str(NEG50_valid))
    NEG100_valid = extract_tensor_value(str(NEG100_valid))

    print(f"\nResults for fold {fold}:")
    print(LOSS_valid, PEARSON_valid, SPEARMAN_valid, NDCG_mean_valid,
          PO10_valid, PO50_valid, PO100_valid,
          NEG10_valid, NEG50_valid, NEG100_valid)

    with open(out_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            f'fold_{fold}', LOSS_valid, PEARSON_valid, SPEARMAN_valid,
            NDCG_mean_valid, PO10_valid, PO50_valid, PO100_valid,
            NEG10_valid, NEG50_valid, NEG100_valid
        ])



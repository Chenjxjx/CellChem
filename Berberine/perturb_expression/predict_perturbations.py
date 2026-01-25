import os
import sys
import json
import yaml
import torch
import numpy as np
import hdf5plugin
import pandas as pd
import anndata
import argparse 
import types
import math
from tqdm import tqdm
from torch import nn
from torch_geometric.data import DataLoader 
from scipy.sparse import issparse

# 添加路径
sys.path.append("../../CellChem_generation/")
try:
    from generate import Generation
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from Model_mol.graphtransformer_molclr import GraphTransformer
    from sm_perturb_ft import molgraph_tokenize, sm_tokenize_and_pad_batch, clue_binning
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# ==============================================================================
# CONFIG 
# ==============================================================================
CONFIG = {
    "device": None, 
    "batch_size": 128, 
    "model_seq_len": 979, 
    "n_bins": 51,
    "cell_data_path": "../../source/clue_cp_level5_prepared_with_cell_rep_generate_random_train.h5ad",
    "smiles_file": "all_target_smiles.json",
    "output_dir": "perturbation_results_final",
    "pretrain_dir": "../../CellChem_pretrain/CellChem/save/dev_clue-May20-00-03/", 
    "finetune_dir": "../../CellChem_generation/save/dev_clue-Jan09-13-48/", 
    "mol_model_dir": "../../CellChem_pretrain/mol_gt_pretrain/ckpt/May08_16-21-26/checkpoints",
}

# ==============================================================================
# Patched Forward
# ==============================================================================
def patched_forward(self, src, values, src_key_padding_mask, data):
    cell_embedding = self.scgpt._encode(
            src,
            values, 
            src_key_padding_mask=src_key_padding_mask,
        ) 
    
    ris, zis = self.mol_encoder(
        data.x, data.batch, 
        data.edge_index, data.edge_attr,
        data.get('chem_desc', None)
    )

    gamma = self.gamma_proj(ris).unsqueeze(1).expand(-1, cell_embedding.size(1), -1)
    beta = self.beta_proj(ris).unsqueeze(1).expand(-1, cell_embedding.size(1), -1)
    x = gamma * cell_embedding + beta 

    q = self.q_proj(x)
    k = self.k_proj(ris).unsqueeze(1)
    v = self.v_proj(ris).unsqueeze(1)
    
    attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    attn_weights = torch.softmax(attn_scores, dim=1)
    attended = attn_weights * v.expand(-1, x.size(1), -1)
    fused = self.fuse_ln(x + attended)

    pred = self.pred_head(fused).squeeze(-1)
    return cell_embedding, ris, pred

# ==============================================================================
# Helper
# ==============================================================================
def load_matched_weights(model, checkpoint_path, device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
        
    model_state = model.state_dict()
    new_state_dict = {}

    for k, v in checkpoint.items():
        name = k.replace("module.", "")
        if name in model_state:
            if v.shape == model_state[name].shape:
                new_state_dict[name] = v
        
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Load result - Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    return model

# ==============================================================================
# Model Initialization
# ==============================================================================
def load_model(config, vocab, fold_num, dtype):
    print(f"\n>>> Initializing Model for Fold {fold_num}...")
    device = config["device"]

    # 1. Load scGPT Config
    args_path = os.path.join(config["pretrain_dir"], "args.json")
    with open(args_path, "r") as f:
        scgpt_args = json.load(f)

    model_scGPT = TransformerModel(
        ntoken=len(vocab),
        d_model=scgpt_args["embsize"],
        nhead=scgpt_args["nheads"],
        d_hid=scgpt_args["d_hid"],
        nlayers=scgpt_args["nlayers"],
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        n_input_bins=config["n_bins"],
        do_mvc=True,
        do_dab=True,
        use_batch_labels=False,
        num_batch_labels=1,
        use_fast_transformer=True 
    )

    # 2. Load GraphTransformer Config
    mol_config_path = os.path.join(config["mol_model_dir"], "config_mg.yaml")
    mol_config = yaml.load(open(mol_config_path, "r"), Loader=yaml.FullLoader)
    graphtransformer = GraphTransformer(**mol_config["model"])

    # 3. Assemble
    model = Generation(model_scGPT, graphtransformer)
    
    # 4. Load Weights
    ckpt_filename = f"best_model_{fold_num}fold.pt"
    ckpt_path = os.path.join(config["finetune_dir"], ckpt_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    model = load_matched_weights(model, ckpt_path, "cpu")
    
    model.forward = types.MethodType(patched_forward, model)
    model.to(device=device, dtype=dtype)
    
    param_device = next(model.parameters()).device
    print(f">>> [Check] Model first parameter is on: {param_device}")
    if str(param_device).split(":")[0] != "cuda":
        raise RuntimeError(f"Model failed to move to CUDA! Current device: {param_device}")

    model.eval()
    return model

# ==============================================================================
# Data Prep (Same as before)
# ==============================================================================
def get_cell_embeddings(adata, vocab, config):
    print("Processing cell data...")
    all_genes = adata.var.index.tolist()
    valid_indices = [i for i, gene in enumerate(all_genes) if gene in vocab]
    
    if len(valid_indices) < len(all_genes):
        print(f"  [Auto-Slice] Filtering genes to {len(valid_indices)}")
        adata = adata[:, valid_indices].copy()
        if 'cell_rep' in adata.obsm:
            X_full = adata.obsm['cell_rep']
            if X_full.shape[1] == len(all_genes):
                adata.obsm['cell_rep'] = X_full[:, valid_indices]

    actual_seq_len = adata.n_vars + 1
    
    if 'cell_rep' in adata.obsm:
        X = adata.obsm['cell_rep']
    else:
        X = adata.X
        if issparse(X): X = X.toarray()

    if "cell_iname" in adata.obs:
        df = pd.DataFrame(X)
        df['type'] = adata.obs['cell_iname'].values
        df_mean = df.groupby('type').mean()
        cell_types = df_mean.index.tolist()
        X_mean = df_mean.values
    else:
        X_mean = X
        cell_types = adata.obs_names.tolist()

    global_min = X.min(axis=0)
    global_max = X.max(axis=0)
    X_combined = np.vstack([X_mean, global_min, global_max])
    
    adata_temp = anndata.AnnData(X=X_combined)
    clue_binning(adata_temp, key_to_process='X', result_binned_key='X_binned', n_bins=config["n_bins"])
    X_binned = adata_temp.layers['X_binned'][:-2]
    
    genes = adata.var.index.tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    
    tokenized = sm_tokenize_and_pad_batch(
        X_binned,
        gene_ids,
        np.zeros(len(X_binned), dtype=int),
        max_len=actual_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=True
    )
    return tokenized, cell_types

class PerturbationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_cells, smiles_list):
        self.genes = tokenized_cells["genes"]
        self.values = tokenized_cells["values"]
        self.smiles_list = smiles_list
        self.n_cells = self.genes.shape[0]
        self.n_mols = len(smiles_list)
    def __len__(self): return self.n_cells * self.n_mols
    def __getitem__(self, idx):
        cell_idx = idx // self.n_mols
        mol_idx = idx % self.n_mols
        smi = self.smiles_list[mol_idx]
        graph = molgraph_tokenize(smi)
        return {
            "gene_ids": self.genes[cell_idx],
            "values": self.values[cell_idx],
            "graph": graph,
            "smiles": smi,
            "cell_idx": cell_idx,
            "mol_idx": mol_idx
        }

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("\n!!!!!!!!!!!!!!! FATAL ERROR !!!!!!!!!!!!!!!")
        print("torch.cuda.is_available() returned False.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        sys.exit(1)
    
    CONFIG["device"] = "cuda"
    print(f">>> CUDA Available: True. Device Count: {torch.cuda.device_count()}")
    print(f">>> Current Device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")

    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print(">>> Using bfloat16")
    else:
        dtype = torch.float16
        print(">>> Using float16")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    vocab_file = os.path.join(CONFIG["finetune_dir"], "vocab_all.json")
    if not os.path.exists(vocab_file):
        vocab_file = os.path.join(CONFIG["pretrain_dir"], "vocab.json")
    vocab = GeneVocab.from_file(vocab_file)
    vocab.set_default_index(vocab["<pad>"])

    model = load_model(CONFIG, vocab, args.fold, dtype)

    adata = anndata.read_h5ad(CONFIG['cell_data_path'])
    tokenized_cells, cell_types = get_cell_embeddings(adata, vocab, CONFIG)
    
    with open(CONFIG["smiles_file"], 'r') as f:
        data = json.load(f)
        if isinstance(data, dict):
            smiles_list = data.get("smiles_list", list(data.values())[0])
        else:
            smiles_list = data
    
    dataset = PerturbationDataset(tokenized_cells, smiles_list)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    print(f"Starting Inference...")
    n_cells = len(cell_types)
    n_mols = len(smiles_list)
    n_genes = tokenized_cells["genes"].shape[1]
    predictions = np.zeros((n_cells, n_mols, n_genes), dtype=np.float16) 

    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                for batch_idx, batch in enumerate(tqdm(dataloader)):
                    gene_ids = batch["gene_ids"].to("cuda", non_blocking=True)
                    values = batch["values"].to("cuda", non_blocking=True).to(dtype)
                    graphs = batch["graph"].to("cuda", non_blocking=True)
                    
                    src_key_padding_mask = gene_ids.eq(vocab["<pad>"])
                    
                    if batch_idx == 0:
                        print(f"\n[DEBUG] Input Tensor Check:")
                        print(f"  gene_ids device: {gene_ids.device}")
                        print(f"  values device: {values.device}, dtype: {values.dtype}")
                        print(f"  Model parameter device: {next(model.parameters()).device}")
                        if str(gene_ids.device).split(":")[0] != 'cuda':
                             raise RuntimeError("Input tensor is NOT on CUDA!")

                    _, _, pred = model(
                        src=gene_ids,
                        values=values,
                        src_key_padding_mask=src_key_padding_mask,
                        data=graphs
                    )
                    
                    if torch.isnan(pred).any():
                        print("[Warning] NaN detected!")

                    pred_np = pred.float().cpu().numpy()
                    c_idxs = batch["cell_idx"].numpy()
                    m_idxs = batch["mol_idx"].numpy()

                    for i in range(len(c_idxs)):
                        predictions[c_idxs[i], m_idxs[i], :] = pred_np[i]

    except Exception as e:
        print("\n!!! Inference Failed !!!")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    save_path = os.path.join(CONFIG["output_dir"], f"predictions_fold{args.fold}.npz")
    print(f"Saving to {save_path}...")
    np.savez_compressed(
        save_path,
        pred=predictions,
        cell_types=cell_types,
        smiles=smiles_list,
        gene_ids=tokenized_cells["genes"][0].numpy()
    )
    print("Done.")

if __name__ == "__main__":
    main()

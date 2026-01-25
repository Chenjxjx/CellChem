import hdf5plugin # 必须最先导入
import os
import sys
import yaml
import torch
import pickle
import pandas as pd
import numpy as np
import anndata
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.append("../../CellChem_generation")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from Model_mol.graphtransformer import GraphTransformer
from sm_perturb_ft import molgraph_tokenize

CONFIG = {
    "adata_path": "../../data/clue_cp_level5_prepared.h5ad",
    "ckpt_path": "../save/dev_clue-May20-00-03/best_model.pt",
    "mol_config_path": "../../mol_gt_pretrain/ckpt/May08_16-21-26/checkpoints/config_mg.yaml",
    "output_dir": "../save/embeddings_output",
    "batch_size": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "new_molecules": {
        "Berberine": "COc1cc2c(cc1OC)[n+]1cc3cc4c(cc3c1Cc2)OCO4",
    }
}

def check_invalid_smiles(smi):
    smi = str(smi)
    if smi in ['nan', 'restricted']:
        return False
    mol = Chem.MolFromSmiles(smi)
    return mol is not None

def preprocess_smiles_list(smiles_list):
    valid_smiles = []
    print(f"Preprocessing {len(smiles_list)} molecules (Validity Check Only - KEEPING SALTS)...")

    for smi in tqdm(smiles_list):
        smi_str = str(smi)
        if len(smi_str) < 2 or smi_str.lower() == 'nan':
            continue

        try:
            mol = Chem.MolFromSmiles(smi_str)
            if mol is None:
                continue
            clean_smi = Chem.MolToSmiles(mol, canonical=True)
            valid_smiles.append(clean_smi)

        except Exception:
            continue

    valid_smiles = sorted(list(set(valid_smiles)))
    print(f"Preprocessing done. Unique valid molecules: {len(valid_smiles)}")
    return valid_smiles

class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
    def __len__(self): return len(self.smiles_list)
    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        try:
            graph = molgraph_tokenize(smi)
            return {"graph": graph, "smiles": smi, "valid": True}
        except:
            return {"graph": None, "smiles": smi, "valid": False}

def collate_fn(batch):
    batch = [b for b in batch if b["valid"] and b["graph"] is not None]
    if len(batch) == 0: return None
    return batch

def main():
    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print("Initializing GraphTransformer...")
    if not os.path.exists(CONFIG["mol_config_path"]):
        raise FileNotFoundError(f"Config not found: {CONFIG['mol_config_path']}")
        
    config_mol = yaml.load(open(CONFIG["mol_config_path"], "r"), Loader=yaml.FullLoader)
    graphtransformer = GraphTransformer(**config_mol["model"]).to(device)

    print(f"Loading weights from {CONFIG['ckpt_path']} ...")
    if not os.path.exists(CONFIG["ckpt_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {CONFIG['ckpt_path']}")

    model_p = torch.load(CONFIG["ckpt_path"], map_location=device)
    
    mol_state = {}
    print("Extracting mol_encoder weights...")
    for param_tensor in model_p:
        key_name = param_tensor.replace("module.", "")
        
        if 'mol_encoder' in key_name:
            if key_name.startswith("mol_encoder."):
                mol_state.update({key_name[12:]: model_p[param_tensor]})
                
    if len(mol_state) == 0:
        raise RuntimeError("Fatal: No 'mol_encoder' keys found. Check checkpoint file structure!")
    
    graphtransformer.load_state_dict(mol_state)
    graphtransformer.eval()
    print(f"Successfully loaded {len(mol_state)} keys into GraphTransformer.")

    print("Loading reference data...")
    adata = anndata.read_h5ad(CONFIG['adata_path'])
    unique_smiles = adata.obs['canonical_smiles'].dropna().unique().tolist()
    
    target_smiles = []
    for name, smi in CONFIG["new_molecules"].items():
        target_smiles.append(smi)
    
    all_smiles = preprocess_smiles_list(unique_smiles + target_smiles)

    dataset = SmilesDataset(all_smiles)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)

    embeddings_map = {}
    print("Extracting aligned embeddings...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            
            graphs = [b['graph'] for b in batch]
            smiles_batch = [b['smiles'] for b in batch]
            
            from torch_geometric.data import Batch
            batch_graph = Batch.from_data_list(graphs).to(device)
            
            ris, _ = graphtransformer(batch_graph.x, batch_graph.batch, batch_graph.edge_index, batch_graph.edge_attr, None)
            
            ris = ris.cpu().numpy()
            
            for i, smi in enumerate(smiles_batch):
                embeddings_map[smi] = ris[i].astype(np.float32)

    emb_out_path = Path(CONFIG["output_dir"]) / "reference_embeddings.pkl"
    with open(emb_out_path, 'wb') as f:
        pickle.dump(embeddings_map, f)
    print(f"Embeddings saved to {emb_out_path}")

    print("Generating metadata table...")
    meta_records = []
    
    def agg_info(df):
        res = {}
        for col in ['pert_id', 'moa', 'target']:
            if col in df.columns:
                vals = df[col].dropna().astype(str).unique()
                res[col] = ' | '.join(sorted([v for v in vals if v not in ['nan', '']]))
        return res

    cols = ['canonical_smiles', 'pert_id', 'moa', 'target']

    df_subset = adata.obs[cols].copy()
    for col in df_subset.columns:
        df_subset[col] = df_subset[col].astype(str)
    df_agg = df_subset.groupby('canonical_smiles').agg(
        lambda x: ' | '.join(sorted(set(v for v in x if v != 'nan')))
    ).reset_index()
    valid_clean_set = set(embeddings_map.keys())

    for _, row in tqdm(df_agg.iterrows(), total=len(df_agg), desc="Mapping Metadata"):
        raw_smi = row['canonical_smiles']
        
        try:
            m = Chem.MolFromSmiles(raw_smi)
            if m:
                frags = Chem.GetMolFrags(m, asMols=True)
                if len(frags) > 1: m = max(frags, key=lambda x: x.GetNumAtoms())
                clean_smi = Chem.MolToSmiles(m, canonical=True)
                
                if clean_smi in valid_clean_set:
                    rec = row.to_dict()
                    rec['canonical_smiles'] = clean_smi 
                    rec['original_smiles'] = raw_smi
                    meta_records.append(rec)
        except: pass

    for name, raw_smi in CONFIG["new_molecules"].items():
        try:
            m = Chem.MolFromSmiles(raw_smi)
            if m:
                frags = Chem.GetMolFrags(m, asMols=True)
                if len(frags) > 1: m = max(frags, key=lambda x: x.GetNumAtoms())
                clean_smi = Chem.MolToSmiles(m, canonical=True)
                
                if clean_smi in valid_clean_set:
                    meta_records.append({
                        'canonical_smiles': clean_smi,
                        'pert_id': f"NEW_{name}",
                        'moa': f"Target Molecule ({name})",
                        'target': 'Unknown',
                        'original_smiles': raw_smi
                    })
        except: pass

    df_out = pd.DataFrame(meta_records)
    meta_out_path = Path(CONFIG["output_dir"]) / "reference_metadata.csv"
    df_out.to_csv(meta_out_path, index=False)
    
    print(f"Metadata saved. Total records: {len(df_out)}")

if __name__ == "__main__":
    main()

import os
import hdf5plugin
import json
import pandas as pd
import anndata
from rdkit import Chem
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Path to the FULL dataset (ensure this contains all drugs, not just training split)
    "source_data_path": "../../CellChem_pretrain/data/clue_cp_level5_prepared.h5ad", 
    
    # Output Directory
    "output_dir": ".",
    "metadata_save_dir": "./save/embeddings_output", # Directory for analysis metadata
    
    # Custom Molecules to inject
    "custom_molecules": {
        "Berberine": {
            "smiles": "COc1cc2c(cc1OC)[n+]1cc3cc4c(cc3c1Cc2)OCO4",
            "pert_id": "NEW_Berberine",
            "moa": "Target Molecule (Berberine)",
            "target": "Topoisomerase | IKK"
        },
        "Doxorubicin": {
            "smiles": "OCC(=O)[C@@]1(O)C[C@H](O[C@H]2C[C@H](N)[C@@H]([C@@H](O2)C)O)c2c(C1)c(O)c1c(c2O)C(=O)c2c(C1=O)cccc2OC",
            "pert_id": "Doxorubicin",
            "moa": "Topoisomerase inhibitor",
            "target": "Topoisomerase"
        }
    }
}

def standardize_smiles_strict(smi):
    """
    Strict standardization logic:
    1. Desalt (split by '.' and take largest fragment).
    2. Canonicalize using RDKit with isomericSmiles=True.
    """
    if not smi or pd.isna(smi) or str(smi) in ['nan', 'restricted']: 
        return None
    try:
        smi_str = str(smi)
        # 1. Desalt
        if '.' in smi_str:
            fragments = smi_str.split('.')
            largest_frag = max(fragments, key=len)
            smi_to_use = largest_frag
        else:
            smi_to_use = smi_str
            
        # 2. Canonicalize
        mol = Chem.MolFromSmiles(smi_to_use)
        if mol: 
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except: 
        pass
    return None

def main():
    # 1. Load Data
    print(f"Loading source data from {CONFIG['source_data_path']}...")
    if not os.path.exists(CONFIG['source_data_path']):
        # Fallback check
        alt_path = "../source/data/clue_cp_level5_prepared.h5ad"
        if os.path.exists(alt_path):
            CONFIG['source_data_path'] = alt_path
            print(f"Using alternative path: {alt_path}")
        else:
            print("Error: Source data file not found.")
            return

    try:
        adata = anndata.read_h5ad(CONFIG['source_data_path'], backed='r')
        df = adata.obs[['canonical_smiles', 'pert_id', 'moa', 'target']].copy()
    except:
        adata = anndata.read_h5ad(CONFIG['source_data_path'])
        df = adata.obs[['canonical_smiles', 'pert_id', 'moa', 'target']].copy()

    print(f"Processing {len(df)} records...")

    # 2. Standardize SMILES (Strict)
    tqdm.pandas(desc="Standardizing SMILES")
    df['clean_smiles'] = df['canonical_smiles'].progress_apply(standardize_smiles_strict)
    
    # Drop invalid entries
    df_clean = df.dropna(subset=['clean_smiles'])
    
    # 3. Aggregate Metadata
    print("Aggregating metadata by clean SMILES...")
    
    def agg_func(x):
        # Unique, sorted, non-empty strings joined by ' | '
        vals = set(str(v) for v in x if str(v).lower() != 'nan' and str(v) != '')
        return ' | '.join(sorted(list(vals)))

    meta_df = df_clean.groupby('clean_smiles').agg({
        'pert_id': agg_func,
        'moa': agg_func,
        'target': agg_func,
        'canonical_smiles': 'first' # Keep one original raw SMILES for reference
    }).reset_index()
    
    # Rename for consistency
    meta_df = meta_df.rename(columns={
        'canonical_smiles': 'original_smiles', 
        'clean_smiles': 'canonical_smiles' # This will be the key
    })

    # 4. Inject Custom Molecules
    print("Injecting custom molecules...")
    new_rows = []
    existing_smiles = set(meta_df['canonical_smiles'])
    custom_map = {} # For JSON output

    for name, info in CONFIG["custom_molecules"].items():
        clean_smi = standardize_smiles_strict(info['smiles'])
        
        if clean_smi:
            custom_map[name] = clean_smi
            
            # Only add to metadata if not present (or force add if you prefer)
            if clean_smi not in existing_smiles:
                new_rows.append({
                    'canonical_smiles': clean_smi,
                    'pert_id': info['pert_id'],
                    'moa': info['moa'],
                    'target': info['target'],
                    'original_smiles': info['smiles']
                })
                print(f"  + Added Metadata for {name}")
            else:
                print(f"  * Metadata for {name} already exists.")
        else:
            print(f"  x Failed to standardize {name}")

    if new_rows:
        meta_df = pd.concat([meta_df, pd.DataFrame(new_rows)], ignore_index=True)

    # 5. Save Outputs
    
    # A. Save JSON list for Prediction
    print("Saving SMILES list for prediction...")
    final_smiles_list = sorted(meta_df['canonical_smiles'].unique().tolist())
    
    json_output = {
        "smiles_list": final_smiles_list,
        "custom_molecules_map": custom_map
    }
    
    json_path = os.path.join(CONFIG["output_dir"], "all_target_smiles.json")
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"  -> Saved {json_path} ({len(final_smiles_list)} molecules)")

    # B. Save CSV for Analysis
    print("Saving Metadata CSV for analysis...")
    os.makedirs(CONFIG["metadata_save_dir"], exist_ok=True)
    csv_path = os.path.join(CONFIG["metadata_save_dir"], "reference_metadata.csv")
    meta_df.to_csv(csv_path, index=False)
    print(f"  -> Saved {csv_path} ({len(meta_df)} records)")

if __name__ == "__main__":
    main()

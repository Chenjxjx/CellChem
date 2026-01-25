import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

INPUT_DIR = "../save/embeddings_output"
TARGET_NAME = "Berberine"
OUTPUT_FILE = "berberine_pearson_rank.csv"

def main():
    print("Loading embeddings and metadata...")
    pkl_path = os.path.join(INPUT_DIR, "mol_reference_embeddings.pkl")
    meta_path = os.path.join(INPUT_DIR, "mol_reference_metadata.csv")
    
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        embed_dict = pickle.load(f)
        
    meta_df = pd.read_csv(meta_path)
    
    target_pert_id_key = f"NEW_{TARGET_NAME}"
    target_row = meta_df[meta_df['pert_id'] == target_pert_id_key]
    
    if len(target_row) == 0:
        print(f"Warning: Could not find '{target_pert_id_key}' in metadata. Trying to search by name in dictionary keys...")
        potential_keys = [k for k in embed_dict.keys() if TARGET_NAME in str(k)]
        target_row = meta_df[meta_df['moa'].fillna('').str.contains(f"Target Molecule \({TARGET_NAME}\)")]

    if len(target_row) == 0:
        print("Error: Target molecule not found in metadata.")
        return

    target_smiles = target_row.iloc[0]['canonical_smiles']
    print(f"Target SMILES: {target_smiles}")
    
    if target_smiles not in embed_dict:
        print("Error: SMILES not found in embedding dictionary.")
        return

    target_vec = embed_dict[target_smiles]
    
    candidate_smiles = []
    candidate_vecs = []
    
    for smi, vec in embed_dict.items():
        if smi == target_smiles: continue
        candidate_smiles.append(smi)
        candidate_vecs.append(vec)
        
    candidate_matrix = np.array(candidate_vecs)
    target_vec = target_vec.reshape(1, -1)
    target_centered = target_vec - target_vec.mean(axis=1, keepdims=True)
    candidate_centered = candidate_matrix - candidate_matrix.mean(axis=1, keepdims=True)
    pearson_scores = cosine_similarity(target_centered, candidate_centered)[0]
    
    df_scores = pd.DataFrame({
        'smiles': candidate_smiles,
        'pearson': pearson_scores
    })
    meta_simple = meta_df[['canonical_smiles', 'pert_id']].drop_duplicates(subset=['canonical_smiles'])
    result_df = df_scores.merge(meta_simple, left_on='smiles', right_on='canonical_smiles', how='left')
    result_df = result_df[['pert_id', 'smiles', 'pearson']]
    result_df = result_df.sort_values('pearson', ascending=False)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved ranking to {OUTPUT_FILE}")
    print("\n--- Top 10 Results ---")
    print(result_df.head(10))

if __name__ == "__main__":
    main()

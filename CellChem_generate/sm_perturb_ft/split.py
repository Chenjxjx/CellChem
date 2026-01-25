import numpy as np
from sklearn.model_selection import KFold

def random_k_fold_split(
    all_counts, 
    control_counts, 
    celltypes_labels, 
    batch_ids, 
    cond_ids, 
    smiles_data=None, 
    shuffle=True, 
    k=5
):
    """
    Performs K-Fold splitting on multiple parallel arrays synchronously.
    """
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=42 if shuffle else None)
    
    folds = []
    n_samples = len(all_counts)

    # Iterate through splits
    for train_idx, valid_idx in kf.split(range(n_samples)):
        
        # Helper for handling optional smiles_data
        if smiles_data is not None:
            train_smiles = smiles_data[train_idx]
            valid_smiles = smiles_data[valid_idx]
        else:
            train_smiles, valid_smiles = None, None

        # Pack the tuple strictly matching the unpacking order in your snippet
        fold_data = (
            all_counts[train_idx],      all_counts[valid_idx],
            control_counts[train_idx],  control_counts[valid_idx],
            celltypes_labels[train_idx], celltypes_labels[valid_idx],
            batch_ids[train_idx],       batch_ids[valid_idx],
            cond_ids[train_idx],        cond_ids[valid_idx],
            train_smiles,               valid_smiles
        )
        folds.append(fold_data)
        
    return folds

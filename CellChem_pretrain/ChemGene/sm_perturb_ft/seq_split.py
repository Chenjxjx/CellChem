import numpy as np
from typing import List

from collections import defaultdict

def generate_clust_mapping(
        clust_label: np.ndarray,
):
    mmp = defaultdict(list)
    for ind, clust_id in enumerate(clust_label):
        mmp[clust_id].append(ind)

    mmp = {key: sorted(value) for key, value in mmp.items()}
    mmp = [
        clust_set for (clust_id, clust_set) in sorted(
            mmp.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    return mmp

def seq_split(dataset, clust_label, valid_size, test_size):
    train_size = 1.0 - valid_size - test_size
    clust_sets = generate_clust_mapping(clust_label)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    for clust_set in clust_sets:
        if len(train_inds) + len(clust_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(clust_set) > valid_cutoff:
                test_inds += clust_set
            else:
                valid_inds += clust_set
        else:
            train_inds += clust_set
    return train_inds, valid_inds, test_inds


def train_test_split_by_seqid(
        *args,
        seq_data,
        clust_label,
        test_size=0.1,
        shuffle=True,
):
    train_inds, valid_inds, test_inds = seq_split(
        seq_data, clust_label, valid_size=test_size, test_size=0,
    )

    if shuffle:
        np.random.shuffle(train_inds)

    outputs = []
    for ag in args + (seq_data, ):
        outputs.extend([ag[train_inds], ag[valid_inds]])

    return outputs

from typing import Dict, Optional, Union

import numpy as np
import torch
from scipy.sparse import issparse
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits

def clue_binning(
        adata: AnnData,
        key_to_process: str = 'X',
        result_binned_key: str = 'X_binned',
        n_bins: int = 51,
):
    print("Binning data ...")
    if not isinstance(n_bins, int):
        raise ValueError(
            "n_bins arg must be an integer, but got {}.".format(n_bins)
        )
    # preliminary checks, will use later
    if key_to_process == "X":
        key_to_process = None  # the following scanpy apis use arg None to use X

    binned_rows = []
    bin_edges = []
    layer_data = _get_obs_rep(adata, layer=key_to_process)
    layer_data = layer_data.A if issparse(layer_data) else layer_data
    # if layer_data.min() < 0:
    #     raise ValueError(
    #         f"Assuming non-negative data, but got min value {layer_data.min()}."
    #     )

    max_val = layer_data.max()
    min_val = layer_data.min()

    for row in layer_data:
        bins = np.linspace(min_val, max_val, n_bins)
        digits = _digitize(row, bins)
        # no zero digits!
        assert digits.min() >= 1
        assert digits.max() <= n_bins
        binned_rows.append(digits)
        bin_edges.append(bins)
    adata.layers[result_binned_key] = np.stack(binned_rows)
    adata.obsm["bin_edges"] = np.stack(bin_edges)

def clue_binning_control(
        adata: AnnData,
        key_to_process: str = 'cell_rep',
        result_binned_key: str = 'X_binned_cell',
        n_bins: int = 51,
):
    print("Binning data ...")
    if not isinstance(n_bins, int):
        raise ValueError(
            "n_bins arg must be an integer, but got {}.".format(n_bins)
        )
    # preliminary checks, will use later
    binned_rows = []
    bin_edges = []
    layer_data = adata.obsm[key_to_process]
    layer_data = layer_data.A if issparse(layer_data) else layer_data
    max_val = layer_data.max()
    min_val = layer_data.min()

    for row in layer_data:
        bins = np.linspace(min_val, max_val, n_bins)
        digits = _digitize(row, bins)
        assert digits.min() >= 1
        assert digits.max() <= n_bins
        binned_rows.append(digits)
        bin_edges.append(bins)
    adata.layers[result_binned_key] = np.stack(binned_rows)
    adata.obsm["bin_edges"] = np.stack(bin_edges)

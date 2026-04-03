import os
from typing import Optional, Union
from pathlib import Path
from functools import reduce
from operator import or_
import numpy as np
import pandas as pd

import cmapPy
import cmapPy.pandasGEXpress.GCToo as GCToo
from cmapPy.pandasGEXpress.parse import parse
import anndata
from anndata import AnnData

this_script = os.path.abspath(__file__)
this_script_dir = os.path.dirname(this_script)
DATA_DIR = Path(os.path.join(this_script_dir, "../../data"))


def parse_df(
        csv_path: Union[str, Path],
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, sep='\t')
    return df

def parse_geneinfo(
        geneinfo_txt: Union[str, Path] = DATA_DIR / 'geneinfo_beta.txt',
        query_feat_space: Optional[str] = 'landmark',
) -> pd.DataFrame:
    df = parse_df(geneinfo_txt)

    if query_feat_space is not None:
        df = df.query(f'feature_space == "{query_feat_space}"')
        df.reset_index(drop=True, inplace=True)

    df['gene_id'] = df['gene_id'].astype(str)

    return df

def parse_cp_meta(
        csv_path: Union[str, Path] = DATA_DIR / 'compoundinfo_beta.txt',
) -> pd.DataFrame:
    df = parse_df(csv_path)
    df = df.drop_duplicates(subset = ['pert_id'])
    df.reset_index(drop=True, inplace=True)
    return df

def parse_siginfo(
        csv_path: Union[str, Path] = DATA_DIR / 'siginfo_beta.txt',
) -> pd.DataFrame:
    return parse_df(csv_path)

def parse_gene_seq_map_csv(
        csv_path: Union[str, Path] = DATA_DIR / 'gene_protein_map.csv',
):
    return pd.read_csv(csv_path)

def parse_cp_gctx(
        gctx_path: Union[str, Path] = DATA_DIR / 'level5_beta_trt_cp_n720216x12328.gctx',
        to_adata: bool = True,
):
    gctx_path = Path(gctx_path)
    if not gctx_path.exists():
        raise FileNotFoundError(gctx_path)

    cp_df = parse_cp_meta()
    sig_df = parse_siginfo()
    gene_df = parse_geneinfo()

    merge_cp_sig_innerr = cp_df.merge(sig_df, how = 'inner', on = 'pert_id')

    # 978 x 720216
    adata = parse(str(gctx_path), cid = merge_cp_sig_innerr['sig_id'], rid = gene_df['gene_id'])

    adata.col_metadata_df = merge_cp_sig_innerr.set_index('sig_id')
    adata.row_metadata_df = gene_df.set_index('gene_id')

    if to_adata:
        print('Convert to anndata...')
        # transpose to row as cells and columns as genes
        adata = AnnData(adata.data_df.T, obs=adata.col_metadata_df, var=adata.row_metadata_df)

    return adata

def parse_xpr_gctx(
        gctx_path: Union[str, Path] = DATA_DIR / 'level5_beta_trt_xpr_n142901x12328.gctx',
        to_adata: bool = True,
):
    gctx_path = Path(gctx_path)
    if not gctx_path.exists():
        raise FileNotFoundError(gctx_path)

    gene_seq_map = parse_gene_seq_map_csv(DATA_DIR / 'gene_protein_map_clust.csv')
    gene_seq_map = gene_seq_map[~gene_seq_map['Sequence'].isna()]
    # filter genes mapped to multiple proteins
    invalid_genes = [k for k, v in gene_seq_map['cmap_name'].value_counts().items() if v > 1]
    gene_seq_map = gene_seq_map[~gene_seq_map['cmap_name'].isin(invalid_genes)]
    sig_df = parse_siginfo()
    sig_df = sig_df[sig_df['pert_type'] == 'trt_xpr']
    sig_df = sig_df[~sig_df['cmap_name'].isna()]
    gene_df = parse_geneinfo()

    merge_xpr_sig_innerr = gene_seq_map.merge(sig_df, how='inner', on='cmap_name')

    # 978 x 116147
    adata = parse(str(gctx_path), cid=merge_xpr_sig_innerr['sig_id'], rid=gene_df['gene_id'])

    adata.col_metadata_df = merge_xpr_sig_innerr.set_index('sig_id')
    adata.row_metadata_df = gene_df.set_index('gene_id')

    if to_adata:
        print('Convert to anndata...')
        # transpose to row as cells and columns as genes
        adata = AnnData(adata.data_df.T, obs=adata.col_metadata_df, var=adata.row_metadata_df)

    return adata

def generate_cond_tokens(
        row,
):
    cond_descrip = '{cell_line} - {trt_dose} - {trt_time}'
    return cond_descrip.format(
        cell_line = row['cell_iname'],
        trt_dose = row['pert_idose'],
        trt_time = row['pert_itime'],
    )

def preprocessing(
        adata: AnnData,
):
    if not isinstance(adata, AnnData):
        raise TypeError(f'adata type must be AnnData, but got {type(adata)}')

    adata.var.rename({'gene_symbol': 'gene_symbols'}, axis=1, inplace=True)

    total_data_num = adata.obs.shape[0]
    filter_mask_list = []
    nearest_dose_type_filter_mask = adata.obs['nearest_dose'].isna()
    print('nearest dose type filtering NaN number:', nearest_dose_type_filter_mask.sum())
    filter_mask_list.append(nearest_dose_type_filter_mask)

    pert_dose_unit_filter_mask = adata.obs['pert_dose_unit'].isna()
    print('pert dose unit filtering NaN number:', pert_dose_unit_filter_mask.sum())
    filter_mask_list.append(pert_dose_unit_filter_mask)

    nan_smiles_filter_mask = adata.obs['canonical_smiles'].isna()
    print('SMILES filtering NaN number:', nan_smiles_filter_mask.sum())
    filter_mask_list.append(nan_smiles_filter_mask)

    dmso_smiles_filter_mask = adata.obs['cmap_name_x'] == 'DMSO'
    print('Small molecules filtering DMSO number:', dmso_smiles_filter_mask.sum())
    filter_mask_list.append(dmso_smiles_filter_mask)

    pert_type_filter_mask = adata.obs['pert_type'] != 'trt_cp'
    print('Pert type is not trt_cp number:', pert_type_filter_mask.sum())
    filter_mask_list.append(pert_type_filter_mask)

    multipl_smiles_filter_mask = adata.obs['canonical_smiles'].str.contains('\.').fillna(False)
    print('Small molecules filtering multpliers number:', multipl_smiles_filter_mask.sum())
    filter_mask_list.append(multipl_smiles_filter_mask)

    itime_is_24_or_6h_filter_mask = ~adata.obs['pert_itime'].isin(['24 h', '6 h'])
    print('Treatment time is 24 h, 6 h number:', itime_is_24_or_6h_filter_mask.sum())
    filter_mask_list.append(itime_is_24_or_6h_filter_mask)

    high_redun_mask = adata.obs['pert_idose'].value_counts() > 10000
    high_redun_idoses = adata.obs['pert_idose'].value_counts().loc[high_redun_mask].index
    low_redun_filter_mask = ~adata.obs['pert_idose'].isin(high_redun_idoses).fillna(False) # total 36210, leading to 684006
    print(f'pert dose frequencey is blow 10000 number:', low_redun_filter_mask.sum())
    filter_mask_list.append(low_redun_filter_mask)

    cell_high_redun_mask = adata.obs['cell_iname'].value_counts() > 5000
    high_redun_cells = adata.obs['cell_iname'].value_counts().loc[cell_high_redun_mask].index
    low_redun_cell_filter_mask = ~adata.obs['cell_iname'].isin(high_redun_cells).fillna(False) # total 162010
    print(f'cell frequencey is blow 10000 number:', low_redun_cell_filter_mask.sum())
    filter_mask_list.append(low_redun_cell_filter_mask)

    filter_mask = reduce(or_, filter_mask_list)
    print(f'Final filtered number:', filter_mask.sum(), 'while original data number:', total_data_num)
    print(f'Valid data number:', total_data_num - filter_mask.sum())

    adata_filtered = adata[~filter_mask, :]
    pert_itime_tokens = adata_filtered.obs['pert_itime'].astype('category').cat.categories  # 2
    print(f'pert itime tokens (length={len(pert_itime_tokens)}):', pert_itime_tokens)
    pert_idose_tokens = adata_filtered.obs['pert_idose'].astype('category').cat.categories  # 15
    print(f'pert idose tokens (length={len(pert_idose_tokens)}):', pert_idose_tokens)
    cell_iname_tokens = adata_filtered.obs['cell_iname'].astype('category').cat.categories  # 20
    print(f'cell iname tokens (length={cell_iname_tokens}):', cell_iname_tokens)

    # 6600??!!!  reduce cell to 456 but datapoint is 505052
    adata_filtered.obs['str_cond'] = adata_filtered.obs.apply(generate_cond_tokens, axis=1)

    return adata_filtered

def preprocessing_xpr(
        adata: AnnData,
):
    if not isinstance(adata, AnnData):
        raise TypeError(f'adata type must be AnnData, but got {type(adata)}')

    adata.var.rename({'gene_symbol': 'gene_symbols'}, axis=1, inplace=True)

    adata.obs['str_cond'] = adata.obs['cell_iname']

    return adata

if __name__ == '__main__':
    import hdf5plugin
    import argparse

    parser = argparse.ArgumentParser(description='Pre-processing the clue cp or xpr gctx file to AnnData')
    parser.add_argument('outpath', type=Path)
    parser.add_argument('-xpr', '--crispr', action='store_true', default=False, help='Processing crispr data')
    parser.add_argument('-cp', '--compound', action='store_true', default=False, help='Processing cmp-pertb data')
    args = parser.parse_args()
    
    if args.compound:
        adata = parse_cp_gctx(to_adata=True)
        adata = preprocessing(adata)
        adata.write_h5ad(
            args.outpath,
            compression=hdf5plugin.FILTERS["zstd"],
            compression_opts=hdf5plugin.Zstd(clevel=5).filter_options
        )

        print('Compound Done!')

    if args.crispr:
        adata = parse_xpr_gctx(to_adata=True)
        adata = preprocessing_xpr(adata)
        adata.write_h5ad(
            args.outpath.parent / (args.outpath.stem + '_xpr' + args.outpath.suffix) if args.compound else args.outpath,
            compression=hdf5plugin.FILTERS["zstd"],
            compression_opts=hdf5plugin.Zstd(clevel=5).filter_options
        )

        print('xpr Done!')

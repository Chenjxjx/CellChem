import numpy as np
import pandas as pd
import json
import os
import sys
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, pearsonr
from collections import Counter
from rdkit import Chem
import gseapy as gp 

sys.path.append("../../CellChem_generation")
try:
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
except ImportError:
    print("Warning: Could not import GeneVocab.")

# ================= Config =================
INPUT_DIR = "perturbation_results_final"
FOLD_INDICES = range(1, 6)
SMILES_JSON = "all_target_smiles.json"
META_PATH = "./save/embeddings_output/reference_metadata.csv"
TARGET_MOL_NAME = "Berberine"
VOCAB_PATH = "../../CellChem_pretrain/CellChem/save/dev_clue-May20-00-03/vocab.json"


TARGET_CELL_INDEX = 9
CELL_LINE_NAME = "HepG2"
SAVE_DIR = f"analysis_{CELL_LINE_NAME}_pub_final"
TMP_DIR = "./tmp_gseapy_cache"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
sns.set_theme(style="ticks", context="paper", font_scale=1.6) 
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

COLOR_UP = '#e74c3c'    
COLOR_DOWN = '#3498db'
COLOR_NS = '#7f8c8d'
TARGET_PATHWAYS = {
    "Cell Cycle": {"genes": ["CCND1", "CDK4", "CDK6", "E2F1", "RB1", "CDKN1A", "PCNA"], "color": "#2ecc71"}, 
    "Mitochondria": {"genes": ["NDUFA", "MT-CO1", "TOMM20", "HSPD1"], "color": "#9b59b6"}, 
    "MAPK/ERK": {"genes": ["MAPK1", "MAPK3", "MAP2K1", "RAF1", "DUSP6"], "color": "#34495e"}, 
    "DNA Repair": {"genes": ["TP53", "BAX", "ATM", "ATR", "BRCA1"], "color": "#f1c40f"} 
}

# ================= Helper Functions =================
def get_canonical(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) if mol else None
    except: return None

def download_library_locally(lib_name, out_dir):
    gmt_file = os.path.join(out_dir, f"{lib_name}.gmt")
    if os.path.exists(gmt_file) and os.path.getsize(gmt_file) > 0:
        return gmt_file
    print(f"Downloading {lib_name}...")
    url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={lib_name}"
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(gmt_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: f.write(chunk)
            return gmt_file
    except Exception: pass
    return None

def standardize_smiles_column(df, col_name):
    canonical_list = []
    for s in df[col_name]:
        can = get_canonical(str(s))
        canonical_list.append(can if can else str(s))
    return canonical_list

def load_metadata_clean(meta_path):
    if not os.path.exists(meta_path): return None
    try:
        df = pd.read_csv(meta_path)
        smi_col = next((c for c in ['canonical_smiles', 'smiles', 'SMILES'] if c in df.columns), None)
        moa_col = next((c for c in ['moa', 'MOA', 'mechanism'] if c in df.columns), None)
        
        if smi_col and moa_col:
            df['join_key'] = standardize_smiles_column(df, smi_col)
            return df[['join_key', moa_col]].rename(columns={moa_col: 'moa'})
    except Exception: pass
    return None

# ================= Data Loading =================
def load_all_drugs_mean(cell_idx, vocab):
    print("Loading ALL drugs data...")
    fold_preds = []
    gene_names = None
    smiles_list = None

    for k in FOLD_INDICES:
        fname = os.path.join(INPUT_DIR, f"predictions_fold{k}.npz")
        if not os.path.exists(fname): continue
        data = np.load(fname, allow_pickle=True)
        key = 'pred' if 'pred' in data else 'predictions'

        if smiles_list is None:
            smiles_list = [get_canonical(s) for s in data['smiles'].tolist()]
        
        if gene_names is None:
            raw_ids = data['gene_ids'][1:]
            try:
                if hasattr(vocab, 'id2token'): gene_names = [vocab.id2token.get(i, f"Unk_{i}") for i in raw_ids]
                elif hasattr(vocab.vocab, 'lookup_token'): gene_names = [vocab.vocab.lookup_token(int(i)) for i in raw_ids]
                elif hasattr(vocab.vocab, 'itos'): gene_names = [vocab.vocab.itos[int(i)] for i in raw_ids]
            except: pass
            
        fold_preds.append(data[key][cell_idx, :, 1:])
    
    return gene_names, np.mean(np.array(fold_preds), axis=0), smiles_list

def load_target_folds(target_canon, cell_idx):
    matrix_target = []
    for k in FOLD_INDICES:
        fname = os.path.join(INPUT_DIR, f"predictions_fold{k}.npz")
        if not os.path.exists(fname): continue
        data = np.load(fname, allow_pickle=True)
        key = 'pred' if 'pred' in data else 'predictions'
        
        s_list = [get_canonical(s) for s in data['smiles'].tolist()]
        try:
            curr_idx = s_list.index(target_canon)
            matrix_target.append(data[key][cell_idx, curr_idx, 1:])
        except ValueError: continue
    return np.array(matrix_target)

# ================= Plotting Functions (PDF Ready) =================

def plot_correlation_dist(corr_df):
    plt.figure(figsize=(10, 8)) 
    sns.histplot(corr_df['correlation'], bins=50, kde=True, color='teal', alpha=0.6, line_kws={'linewidth': 3})
    
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2)
    plt.text(0.98, 0.95, "Self (r=1.0)", transform=plt.gca().transAxes, 
             color='red', ha='right', fontsize=16, fontweight='bold')
    known_hits = corr_df.iloc[1:][corr_df.iloc[1:]['moa'] != 'Unknown']
    if not known_hits.empty:
        top_hit = known_hits.iloc[0]
        r_val, moa_str = top_hit['correlation'], top_hit['moa']
        if len(moa_str) > 30: moa_str = moa_str[:28] + "..."
        
        plt.axvline(r_val, color='orange', linestyle='--', linewidth=2)
        plt.text(r_val - 0.02, plt.ylim()[1] * 0.7, 
                 f"Top Known Hit:\n{moa_str}\n(r={r_val:.2f})", 
                 color='#d35400', ha='right', fontsize=14, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title(f"Prediction Similarity Distribution\n(Target: {TARGET_MOL_NAME}, Cell Line: {CELL_LINE_NAME})", fontsize=20, pad=20)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "qc_correlation_dist.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def plot_moa_prediction(corr_df):
    top_n_search = 200
    subset = corr_df.iloc[1:top_n_search] 
    known_subset = subset[subset['moa'] != 'Unknown']
    
    if known_subset.empty: return
    
    moa_list = []
    for m in known_subset['moa']:
        split_moas = [x.strip() for x in str(m).replace('|', ',').split(',')]
        moa_list.extend(split_moas)
    
    counts = Counter(moa_list).most_common(10)
    moas, nums = zip(*counts)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=list(nums), y=list(moas), 
        hue=list(moas), palette="viridis", 
        edgecolor='black', legend=False
    )
    
    plt.title(f"Top Predicted Mechanisms (MOA)\n(Analyzed from Top Known Similar Drugs)\n(Target: {TARGET_MOL_NAME}, Cell Line: {CELL_LINE_NAME})", fontsize=20, pad=20)
    plt.xlabel(f"Frequency (in top {len(known_subset)} matched hits)", fontsize=18)
    plt.ylabel("")
    
    for i, v in enumerate(nums):
        plt.text(v + 0.1, i, str(v), color='black', va='center', fontweight='bold', fontsize=14)
        
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "moa_prediction.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def plot_robust_volcano_pub(df):
    plt.figure(figsize=(12, 10))
    
    # 背景点 (深灰，半透明)
    bg_df = df[(df['mean_score'] > -4) & (df['mean_score'] < 5)]
    sns.scatterplot(
        data=bg_df, x='mean_score', y='neg_log_p', 
        color=COLOR_NS, alpha=0.5, s=25, linewidth=0, 
        label='Background'
    )
    
    texts = []
    for pathway, info in TARGET_PATHWAYS.items():
        subset = df[df['gene'].isin(info['genes'])].copy()
        if not subset.empty:
            plt.scatter(
                subset['mean_score'], subset['neg_log_p'],
                c=info['color'], s=90, alpha=1.0, 
                edgecolors='white', linewidth=1,
                label=pathway, zorder=3
            )
            subset['abs_score'] = subset['mean_score'].abs()
            top_genes = subset.sort_values('abs_score', ascending=False).head(3)
            for _, row in top_genes.iterrows():
                texts.append(plt.text(
                    row['mean_score'], row['neg_log_p'], row['gene'], 
                    fontsize=14, fontweight='bold', color=info['color'],
                    ha='right' if row['mean_score'] < 0 else 'left', va='bottom'
                ))

    plt.axvline(0, color='black', lw=1, alpha=0.5)
    plt.axhline(-np.log10(0.05), color='gray', ls='--', lw=1.5, label='P=0.05')
    
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=1))
    except: pass

    plt.title(f"Robustness Volcano: {TARGET_MOL_NAME} (Cell Line: {CELL_LINE_NAME})", fontsize=22, pad=15)
    plt.xlabel("Mean Level 5 Score (Intensity)", fontsize=18)
    plt.ylabel("-log10(P-value) (Consistency)", fontsize=18)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "volcano.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def plot_gsea_bubble_pub(gsea_res):
    if gsea_res.empty: return
    
    top_up = gsea_res[gsea_res['NES'] > 0].head(8) 
    top_down = gsea_res[gsea_res['NES'] < 0].tail(8) 
    plot_df = pd.concat([top_up, top_down]).copy()
    plot_df['Term_Clean'] = plot_df['Term'].apply(lambda x: x.split(' (GO:')[0])

    def abbreviate_term(term):
        term = term.replace('Regulation Of', 'Reg. Of')
        term = term.replace('Positive Regulation Of', 'Pos. Reg. Of')
        term = term.replace('Negative Regulation Of', 'Neg. Reg. Of')
        term = term.replace('Ubiquitin-Dependent Protein Catabolic Process', 'Ubiquitin-Dep. Catabolism')
        return term
    plot_df['Term_Clean'] = plot_df['Term_Clean'].apply(abbreviate_term)
    # -------------------------------------------------------

    plot_df['FDR q-val'] = pd.to_numeric(plot_df['FDR q-val'], errors='coerce')
    plot_df['Significance'] = -np.log10(plot_df['FDR q-val'].fillna(1.0) + 1e-10)
    plt.figure(figsize=(10, 8)) 
    
    norm = plt.Normalize(vmin=-3, vmax=3)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])

    ax = sns.scatterplot(
        data=plot_df, x='NES', y='Term_Clean', size='Significance', 
        sizes=(150, 900),
        hue='NES', palette="RdBu_r", hue_norm=(-2.5, 2.5),
        edgecolor='black', linewidth=1.5
    )

    # 5. 样式调整
    plt.axvline(0, color='black', ls='-', alpha=0.3)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.title(f"GSEA Pathway Enrichment: {TARGET_MOL_NAME} (Cell Line: {CELL_LINE_NAME})", fontsize=24, pad=20)
    plt.xlabel("Normalized Enrichment Score (NES)", fontsize=20)
    plt.ylabel("")
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 6. Colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label("NES", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        title="Significance (-log FDR)", 
        fontsize=14, 
        title_fontsize=16,
        frameon=False,
        labelspacing=1.0
    )

    plt.subplots_adjust(left=0.45, right=0.85, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(SAVE_DIR, "gsea_bubble.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
# ================= Main =================
def main():
    print(f"Loading Vocab from {VOCAB_PATH}...")
    vocab = GeneVocab.from_file(VOCAB_PATH)

    # 1. MOA & Correlation Analysis
    gene_names, all_preds, all_smiles = load_all_drugs_mean(TARGET_CELL_INDEX, vocab)
    
    with open(SMILES_JSON, 'r') as f:
        target_smi_raw = json.load(f).get("custom_molecules_map", {}).get(TARGET_MOL_NAME)
    target_canon = get_canonical(target_smi_raw)
    
    if target_canon not in all_smiles:
        print("Target drug not found.")
        return

    t_idx = all_smiles.index(target_canon)
    target_vec = all_preds[t_idx]
    
    print("Calculating Correlations...")
    corrs = [pearsonr(target_vec, vec)[0] for vec in all_preds]
    
    df_corr = pd.DataFrame({'smiles': all_smiles, 'correlation': corrs})
    
    # Merge Metadata
    meta_df = load_metadata_clean(META_PATH)
    if meta_df is not None:
        df_corr = df_corr.merge(meta_df, left_on='smiles', right_on='join_key', how='left')
    else:
        df_corr['moa'] = 'Unknown'
    
    df_corr['moa'] = df_corr['moa'].fillna('Unknown')
    df_corr = df_corr.sort_values('correlation', ascending=False).reset_index(drop=True)
    
    print("Plotting Correlation & MOA (PDF)...")
    plot_correlation_dist(df_corr)
    plot_moa_prediction(df_corr)

    # 2. Robust Volcano Analysis
    print("Reloading folds for robust statistics...")
    matrix_target = load_target_folds(target_canon, TARGET_CELL_INDEX)
    if matrix_target.size == 0: return

    mean_scores = np.mean(matrix_target, axis=0)
    _, p_val = ttest_1samp(matrix_target, popmean=0, axis=0)
    
    df_res = pd.DataFrame({
        'gene': gene_names,
        'mean_score': mean_scores,
        'pvalue': p_val
    })
    df_res['pvalue'] = df_res['pvalue'].fillna(1.0).replace(0, 1e-300)
    df_res['neg_log_p'] = -np.log10(df_res['pvalue'])
    
    print("Plotting Volcano (PDF)...")
    plot_robust_volcano_pub(df_res)
    
    # 3. GSEA Analysis
    print("Running GSEA...")
    jitter = np.random.normal(0, 1e-6, size=len(df_res))
    df_res['rank_metric'] = df_res['mean_score'] + jitter
    rank_df = df_res[['gene', 'rank_metric']].sort_values('rank_metric', ascending=False)
    
    local_gmt = download_library_locally('GO_Biological_Process_2023', TMP_DIR)
    try:
        pre_res = gp.prerank(rnk=rank_df, gene_sets=local_gmt,
                             threads=4, min_size=15, max_size=1000,
                             permutation_num=1000, outdir=SAVE_DIR, seed=42)
        gsea_res = pre_res.res2d.sort_values('NES', ascending=False)
        print("Plotting GSEA Bubble (PDF)...")
        plot_gsea_bubble_pub(gsea_res)
    except Exception as e:
        print(f"GSEA Failed: {e}")

    print(f"All done. PDF Figures in {SAVE_DIR}")

if __name__ == "__main__":
    main()


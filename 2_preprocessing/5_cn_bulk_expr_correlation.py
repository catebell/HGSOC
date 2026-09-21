from os import path, makedirs
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

import config_file_tree

""""" paths and parameters """""

use_top_genes = True  # if False, uses all genes appearing in RNA matrix

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR

rna_filepath = path.join(preprocessed_dir, "bulk_expr_top_genes_log2.tsv")
top_genes_filepath = path.join(preprocessed_dir, "selected_top_genes.tsv")
cnv_filepath = path.join(preprocessed_dir, "preprocessed_cn.tsv")
metadata_filepath = path.join(preprocessed_dir, "preprocessed_samples_metadata.tsv")

output_dir = path.join(preprocessed_dir, "correlations")
makedirs(output_dir, exist_ok=True)


def clean_sample_id(sample_name):
    """ Remove suffixes "_RNA1/2" and "_DNA1/2" to obtain unique sample ID. """
    return re.sub(r'_(RNA|DNA)\d*$', '', str(sample_name).strip())


def run_cnv_rna_correlation():
    print("Loading rna seq data...")
    top_genes_df = pd.read_csv(top_genes_filepath, sep='\t')

    df_rna = pd.read_csv(rna_filepath, sep='\t', index_col=0)

    if use_top_genes:
        valid_genes = set(top_genes_df['gene_id'])
        df_rna = df_rna[df_rna.index.isin(valid_genes)]
    else:
        valid_genes = df_rna.index

    rna_sample_map = {col: clean_sample_id(col) for col in df_rna.columns}  # clean samples IDs
    df_rna = df_rna.rename(columns=rna_sample_map)
    df_rna = df_rna.T.groupby(level=0).aggregate('mean').T  # eventual duplicated cols

    print("Loading copy numbers data...")
    df_cn = pd.read_csv(cnv_filepath, sep='\t')

    if use_top_genes:
        gene_name_map = dict(zip(top_genes_df['gene_id'], top_genes_df['gene_name']))
    else:
        gene_name_map = dict(zip(df_cn['ID'], df_cn['Gene']))

    # keep only top N genes identified by expr variance
    df_cnv_filtered = df_cn[df_cn['ID'].isin(valid_genes)].copy()
    df_cnv_filtered['sample'] = df_cnv_filtered['sample'].apply(clean_sample_id)
    df_cnv_filtered['tot_copies'] = df_cnv_filtered['nMajor'] + df_cnv_filtered['nMinor']

    # CNV metric chosen to correlate
    cn_values = 'purifiedLogR' # or tot_copies
    df_cnv_pivot = df_cnv_filtered.pivot_table(
        index='ID',
        columns='sample',
        values=cn_values,
        aggfunc='mean'
    )

    common_genes = df_rna.index.intersection(df_cnv_pivot.index)
    common_samples = df_rna.columns.intersection(df_cnv_pivot.columns)

    print(f"Matched Genes: {len(common_genes)} / {len(valid_genes)}")
    print(f"Matched Samples (Patient_Tissue): {len(common_samples)}")

    rna_mat = df_rna.loc[common_genes, common_samples]
    cnv_mat = df_cnv_pivot.loc[common_genes, common_samples]

    """"" computation """""

    print("\nComputing Spearman Correlation between all genes for each sample (n_patients x n_samples)...")
    sample_correlations = rna_mat.corrwith(cnv_mat, axis=0, method='spearman')

    df_sample_corr = pd.DataFrame({
        'sample_id': sample_correlations.index,
        'spearman_rho': sample_correlations.values
    }).dropna(subset=['spearman_rho']).sort_values(by='spearman_rho', ascending=False)

    # analysis by tissue
    if path.exists(metadata_filepath):
        df_meta = pd.read_csv(metadata_filepath, sep='\t')
        df_meta['sample_id'] = df_meta['sample_id'].apply(clean_sample_id)
        df_sample_corr = df_sample_corr.merge(df_meta[['sample_id', 'tissue', 'patient_id']].drop_duplicates(),
                                              on='sample_id', how='left')

    df_sample_corr.to_csv(path.join(output_dir, "cnv_rna_sample_correlations.tsv"), sep='\t', index=False)

    print("\nComputing Spearman Correlation between all genes considering all samples (n_genes x n_samples)...")
    gene_results = []

    for gene in common_genes:
        rna_vals = rna_mat.loc[gene]
        cnv_vals = cnv_mat.loc[gene]

        # Rimuovi eventuali missing values
        mask = rna_vals.notna() & cnv_vals.notna()
        r_clean = rna_vals[mask]
        c_clean = cnv_vals[mask]

        # Verifica varianza minima (evita divisione per zero in serie costanti)
        if len(r_clean) >= 3 and r_clean.nunique() > 1 and c_clean.nunique() > 1:
            rho, pval = spearmanr(r_clean, c_clean)
        else:
            rho, pval = np.nan, np.nan

        gene_results.append({
            'gene_id': gene,
            'gene_name': gene_name_map.get(gene, 'N/A'),
            'spearman_rho': rho,
            'pvalue': pval
        })

    df_gene_corr = pd.DataFrame(gene_results).dropna(subset=['spearman_rho', 'pvalue'])

    # Correzione per test multipli (Benjamini-Hochberg FDR)
    _, fdr_values, _, _ = multipletests(df_gene_corr['pvalue'], method='fdr_bh')
    df_gene_corr['fdr'] = fdr_values

    df_gene_corr = df_gene_corr.sort_values(by='spearman_rho', ascending=False)

    '''
    gene_correlations = rna_mat.corrwith(cnv_mat, axis=1, method='spearman')

    df_gene_corr = pd.DataFrame({
        'gene_id': gene_correlations.index,
        'gene_name': [gene_name_map.get(g, 'N/A') for g in gene_correlations.index],
        'spearman_rho': gene_correlations.values
    }).sort_values(by='spearman_rho', ascending=False)
    '''
    df_gene_corr.to_csv(path.join(output_dir, "cnv_rna_gene_correlations.tsv"), sep='\t', index=False)


    print("\nGraphs creation...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    """"" 1) Correlations distributions for sample (n_patients x n_samples) """""

    if 'tissue' in df_sample_corr.columns:
        sns.boxplot(data=df_sample_corr, x='tissue', y='spearman_rho', ax=axes[0], hue='tissue', palette='Set2', legend=False)
        sns.stripplot(data=df_sample_corr, x='tissue', y='spearman_rho', ax=axes[0], color='black', alpha=0.3,
                      jitter=0.2)
        axes[0].set_title(f"Correlation distribution RNA (expr) - CNV ({cn_values}) by TISSUE")
        axes[0].set_ylabel("Spearman Rho")
    else:
        sns.histplot(df_sample_corr['spearman_rho'], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f"Correlation distribution RNA (expr) - CNV ({cn_values}) by SAMPLE")
        axes[0].set_ylabel("Density / Samples")

    axes[0].set_xlabel("Spearman Rho")
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].grid(True, linestyle='--', alpha=0.5)


    """"" 2) Correlations distributions for gene """""

    sns.histplot(df_gene_corr['spearman_rho'], kde=True, ax=axes[1], color='teal', bins=40)
    axes[1].set_title(f"Correlation distribution RNA (expr) - CNV ({cn_values}) by GENE")
    axes[1].set_xlabel("Spearman Rho (between samples)")
    axes[1].set_ylabel("Genes count")
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_output_path = path.join(output_dir, "cnv_rna_correlation_summary.png")
    plt.savefig(plot_output_path, dpi=300)
    plt.show()

    sig_pos = df_gene_corr[(df_gene_corr['fdr'] < 0.05) & (df_gene_corr['spearman_rho'] > 0)]
    sig_neg = df_gene_corr[(df_gene_corr['fdr'] < 0.05) & (df_gene_corr['spearman_rho'] < 0)]
    uncoupled = df_gene_corr[df_gene_corr['spearman_rho'].abs() < 0.1]
    total_genes = len(df_gene_corr)

    print("\n" + "=" * 60)
    print(
        f"Average correlation per sample: {df_sample_corr['spearman_rho'].mean():.3f} +/- {df_sample_corr['spearman_rho'].std():.3f}")
    print(
        f"Average correlation per gene:   {df_gene_corr['spearman_rho'].mean():.3f} +/- {df_gene_corr['spearman_rho'].std():.3f}")
    print("-" * 60)
    print(f"Total analyzed genes with valid variance: {total_genes}")
    print(
        f"Significantly positively correlated (FDR < 0.05, Rho > 0): {len(sig_pos)} ({len(sig_pos) / total_genes * 100:.1f}%)")
    print(
        f"Significantly negatively correlated (FDR < 0.05, Rho < 0): {len(sig_neg)} ({len(sig_neg) / total_genes * 100:.1f}%)")
    print(f"Genes with uncoupled signal (|Rho| < 0.1): {len(uncoupled)} ({len(uncoupled) / total_genes * 100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_cnv_rna_correlation()
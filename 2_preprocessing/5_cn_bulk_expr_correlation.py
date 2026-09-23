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

use_top_genes = False  # if False, uses all genes appearing in RNA matrix
use_driver_genes = False  # if True, uses known driver genes

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR

rna_filepath = path.join(preprocessed_dir, "bulk_expr_top_genes_log2.tsv")
top_genes_filepath = path.join(preprocessed_dir, "selected_top_genes.tsv")
cnv_filepath = path.join(preprocessed_dir, "preprocessed_cn.tsv")
metadata_filepath = path.join(preprocessed_dir, "preprocessed_samples_metadata.tsv")
known_drivers_filepath = path.join("..", "2_preprocessing", "known_driver_genes", "IntOGen-DriverGenes_OVT.tsv")

output_dir = path.join(preprocessed_dir, "correlations")
makedirs(output_dir, exist_ok=True)


def clean_sample_id(sample_name):
    """ Remove suffixes "_RNA1/2" and "_DNA1/2" to obtain unique sample ID. """
    return re.sub(r'_(RNA|DNA)\d*$', '', str(sample_name).strip())


def compute_sample_correlations(rna_mat, cnv_mat):

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
    return df_sample_corr


def compute_gene_correlations(rna_mat, cnv_mat, common_genes, gene_name_map):
    print("\nComputing Spearman Correlation between all genes considering all samples (n_genes x n_samples)...")
    gene_results = []

    for gene in common_genes:
        rna_vals = rna_mat.loc[gene]
        cnv_vals = cnv_mat.loc[gene]

        # Remove eventual missing values
        mask = rna_vals.notna() & cnv_vals.notna()
        r_clean = rna_vals[mask]
        c_clean = cnv_vals[mask]

        # Verify minimum variance (avoid zero-division in constant series)
        if len(r_clean) >= 3 and r_clean.nunique() > 1 and c_clean.nunique() > 1:
            rho, pval = spearmanr(r_clean, c_clean)
        else:
            rho, pval = np.nan, np.nan

        gene_name_map_inv = dict(zip(gene_name_map.values(), gene_name_map.keys()))

        gene_results.append({
            'gene_id': gene,
            'gene_name': gene_name_map_inv.get(gene, 'N/A'),
            'spearman_rho': rho,
            'pvalue': pval
        })

    df_gene_corr = pd.DataFrame(gene_results).dropna(subset=['spearman_rho', 'pvalue'])

    # Benjamini-Hochberg FDR (False Discovery Rate)
    # FDR corrects raw p-values estimating the % of false-positive results between those declared significant.
    # ex. A value is considered relevant with FDR < 0.05 , meaning it is expected that at most 5% of correlations
    # identified as positive are false.

    _, fdr_values, _, _ = multipletests(df_gene_corr['pvalue'], method='fdr_bh')
    df_gene_corr['fdr'] = fdr_values

    df_gene_corr = df_gene_corr.sort_values(by='spearman_rho', ascending=False)

    df_gene_corr.to_csv(path.join(output_dir, "cnv_rna_gene_correlations.tsv"), sep='\t', index=False)
    return df_gene_corr


def compute_gene_tissue_correlations(rna_mat, cnv_mat, common_genes, common_samples, df_cn):
    """Gene x Tissue correlation between bulk-expr and copy number values """

    if path.exists(metadata_filepath):
        print("\nComputing Spearman Correlation per gene grouped by TISSUE...")
        df_meta = pd.read_csv(metadata_filepath, sep='\t')
        df_meta['sample_id'] = df_meta['sample_id'].apply(clean_sample_id)

        # dict sample_id -> tissue
        sample_to_tissue = df_meta.set_index('sample_id')['tissue'].to_dict()

        # samples mapped to tissue in common
        samples_tissue_series = pd.Series(common_samples).map(sample_to_tissue)
        unique_tissues = [t for t in samples_tissue_series.dropna().unique() if str(t).strip()]

        gene_name_map_inv = dict(zip(df_cn['ID'], df_cn['Gene']))
        tissue_corr_dict = {
            gene: {
                'gene_id': gene,
                'gene_name': gene_name_map_inv.get(gene, 'N/A')
            }
            for gene in common_genes
        }

        # correlation for each tissue
        for tissue in unique_tissues:
            tissue_samples = [s for s in common_samples if sample_to_tissue.get(s) == tissue]

            # at least 5 samples to consider Spearman relevant
            if len(tissue_samples) >= 5:
                rna_t = rna_mat[tissue_samples]
                cnv_t = cnv_mat[tissue_samples]

                for gene in common_genes:
                    r_vals = rna_t.loc[gene]
                    c_vals = cnv_t.loc[gene]

                    mask = r_vals.notna() & c_vals.notna()
                    r_clean = r_vals[mask]
                    c_clean = c_vals[mask]

                    if len(r_clean) >= 3 and r_clean.nunique() > 1 and c_clean.nunique() > 1:
                        rho, _ = spearmanr(r_clean, c_clean)
                    else:
                        rho = np.nan

                    tissue_corr_dict[gene][tissue] = rho

        df_gene_tissue_corr = pd.DataFrame(list(tissue_corr_dict.values()))

        # columns order: gene_id, gene_name, then tissues
        present_tissues = [t for t in unique_tissues if t in df_gene_tissue_corr.columns]
        cols = ['gene_id', 'gene_name'] + present_tissues
        df_gene_tissue_corr = df_gene_tissue_corr[cols]

        tissue_corr_path = path.join(output_dir, "cnv_rna_gene_correlations_by_tissue.tsv")
        df_gene_tissue_corr.to_csv(tissue_corr_path, sep='\t', index=False)
        print(f"Saved tissue-specific correlation matrix to: {tissue_corr_path}")

        """"" Heatmap for first N genes with highest avg correlation """""

        if len(present_tissues) > 1:
            plt.figure(figsize=(8, 10))

            df_plot = df_gene_tissue_corr.copy()
            df_plot['mean_rho'] = df_plot[present_tissues].mean(axis=1)
            top_genes_plot = df_plot.sort_values(by='mean_rho', ascending=False).head(30)

            heatmap_data = top_genes_plot.set_index('gene_name')[present_tissues]

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                        cbar_kws={'label': 'Spearman Rho'})
            plt.title("Top 30 Genes: RNA-CNV Spearman Correlation by Tissue")
            plt.tight_layout()
            plt.savefig(path.join(output_dir, "top_genes_correlation_by_tissue_heatmap.png"), dpi=300)
            plt.close()

        return df_gene_tissue_corr
    return None


def plot_and_summarize_correlations(df_sample_corr, df_gene_corr, cn_values):
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
    print("Loading rna seq data...")
    df_rna = pd.read_csv(rna_filepath, sep='\t', index_col=0)
    print("Loading copy numbers data...")
    df_cn = pd.read_csv(cnv_filepath, sep='\t')

    gene_name_map = dict(zip(df_cn['Gene'].str.upper(), df_cn['ID']))

    if use_driver_genes and path.exists(known_drivers_filepath):
        gene_subset = pd.read_csv(known_drivers_filepath, sep='\t')
    elif use_top_genes and path.exists(top_genes_filepath):
        gene_subset = pd.read_csv(top_genes_filepath, sep='\t')

    if gene_subset is not None:
        if "ENSG" not in str(
                gene_subset.iloc[0, 0]):  # either gene name or gene_id, check first value of first df column
            mapped_symbols = {}
            unmapped = []

            for symbol in gene_subset.iloc[:, 0]:
                ensg = gene_name_map.get(str(symbol).upper())
                if ensg:
                    mapped_symbols[symbol] = ensg
                else:
                    unmapped.append(symbol)

            print(
                f"Successfully mapped: {len(mapped_symbols)} / {len(gene_subset.iloc[:, 0])} genes"
            )
            if unmapped:
                print(f"Genes not found: {unmapped}")

            gene_subset.iloc[:, 0] = gene_subset.iloc[:, 0].map(gene_name_map)
            gene_subset = gene_subset.dropna()
            gene_subset = gene_subset.rename(columns={gene_subset.columns[0]: 'gene_id'})

        valid_genes = set(gene_subset['gene_id'])
        df_rna = df_rna[df_rna.index.isin(valid_genes)]
    else:
        valid_genes = df_rna.index

    rna_sample_map = {col: clean_sample_id(col) for col in df_rna.columns}  # clean samples IDs
    df_rna = df_rna.rename(columns=rna_sample_map)
    df_rna = df_rna.T.groupby(level=0).aggregate('mean').T  # eventual duplicated cols

    # keep only top N genes identified by expr variance
    df_cnv_filtered = df_cn[df_cn['ID'].isin(valid_genes)].copy()
    df_cnv_filtered['sample'] = df_cnv_filtered['sample'].apply(clean_sample_id)
    df_cnv_filtered['tot_copies'] = df_cnv_filtered['nMajor'] + df_cnv_filtered['nMinor']

    # CNV metric chosen to correlate
    cn_values = 'purifiedLogR'  # or tot_copies
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

    df_sample_corr = compute_sample_correlations(rna_mat, cnv_mat)
    df_gene_corr = compute_gene_correlations(rna_mat, cnv_mat, common_genes, gene_name_map)
    compute_gene_tissue_correlations(rna_mat, cnv_mat, common_genes, common_samples, df_cn)
    plot_and_summarize_correlations(df_sample_corr, df_gene_corr, cn_values)
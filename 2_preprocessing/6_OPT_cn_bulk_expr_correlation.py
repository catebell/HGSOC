from os import path, makedirs
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import config_file_tree

""""" paths and parameters """""

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR

rna_filepath = path.join(preprocessed_dir, "preprocessed_bulk_expr.tsv")
cnv_filepath = path.join(preprocessed_dir, "preprocessed_cn.tsv")
metadata_filepath = path.join(preprocessed_dir, "preprocessed_samples_metadata.tsv")

selected_genes_filepath = path.join(preprocessed_dir, "selected_genes.tsv")
known_drivers_filepath = config_file_tree.DRIVER_GENES_FILE

output_dir = path.join(preprocessed_dir, "out_analysis")
makedirs(output_dir, exist_ok=True)


def format_correlation(rho, pval):
    if pd.isna(rho) or pd.isna(pval):
        return ""
    if pval > 0.05:
        stars = " (ignore)"
    else:
        stars = ""
    """
    if pd.isna(rho) or pd.isna(pval):
        return ""
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"
    else:
        stars = ""  # not statistically relevant
    """
    return f"{rho:.3f}{stars}"


def clean_sample_id(sample_name):
    """ Remove suffixes "_RNA1/2" and "_DNA1/2" or other to obtain sample ID. """
    return re.sub(r'_(RNA|DNA)\d*$', '', str(sample_name).strip())


def patient_tissue_level_corr(df_expr, df_cn):
    """ Correlation between bulk-expr array and copy numbers array for every patient, differentiated by tissue. """
    df_metadata = pd.read_csv(metadata_filepath, sep='\t')  # for extracting list of patients and tissues, to visualize what's missing
    df_metadata['sample_id'] = df_metadata['sample_id'].apply(clean_sample_id)

    patients_list = set(df_metadata["patient_id"])
    tissues_list = set(df_metadata["tissue"])

    table = pd.DataFrame(columns=list(tissues_list), index=list(patients_list), dtype=float)

    patients_in_expr = set(col.split("_")[0] for col in df_expr.columns)
    print("\nPatients in expr table: " + str(len(patients_in_expr)))

    patients_in_cn = set(col.split("_")[0] for col in df_cn.columns)
    print("Patients in cn table: " + str(len(patients_in_cn)))

    valid_patients = sorted(list(patients_in_expr.intersection(patients_in_cn)))
    print("Patients in common: " + str(len(valid_patients)))

    duplicates_records = []  # eventual multiple samples from same tissue of same patient
    unrecognized_tissues = []

    print("\nComputing correlations...")

    for p in tqdm(valid_patients):
        # selecting p's columns
        cols_expr = [c for c in df_expr.columns if c.startswith(f"{p}_")]
        cols_cn = [c for c in df_cn.columns if c.startswith(f"{p}_")]

        # keep only samples with both Expr and Copy Number
        common_samples = set(cols_expr).intersection(set(cols_cn))

        seen_tissues = set()
        for sample_key in (sorted(common_samples)):
            parts = sample_key.split("_")
            patient_id = parts[0]
            tissue = parts[1]

            for t in tissues_list:
                if t in tissue:
                    tissue = t
                    break

            if tissue not in tissues_list:
                unrecognized_tissues.append(tissue)
                continue  # skip correlation

            # Spearman corr
            rho, pval = spearmanr(df_expr[sample_key], df_cn[sample_key], nan_policy='omit')

            # if tissue not already found for patient p
            if tissue not in seen_tissues:
                seen_tissues.add(tissue)
                if patient_id in table.index and tissue in table.columns:
                    table.loc[patient_id, tissue] = rho
            else:
                # if tissue already seen for p, new occurrence saved for inspection
                duplicates_records.append({
                    "patient": patient_id,
                    "tissue": tissue,
                    "sample_key": sample_key,
                    "spearman_rho": rho,
                    "pvalue": pval
                })

    df_duplicates = pd.DataFrame(duplicates_records)

    print("--- TABELLA CORRELAZIONE SPEARMAN (Espressione vs CN) ---\n")
    table.to_csv(path.join(output_dir, "corr_expr_cn_patient_tissue_level.tsv"), sep='\t')
    print(table)

    print("\n\n--- MULTIPLE OCCURRENCES OF SAME TISSUE ---\n")
    if not df_duplicates.empty:
        print(df_duplicates)
        df_duplicates.to_csv(path.join(output_dir, "corr_patients_tissue_level_duplicates.tsv"), sep='\t', index=False)
    else:
        print("No duplicated tissue found.")

    print("\n\n--- UNRECOGNIZED TISSUES ---\n")
    if len(unrecognized_tissues) > 0:
        print(unrecognized_tissues)
    else:
        print("No unrecognized tissues found.")


    """"" Heatmap """""

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 12))

    num_patients = len(table)
    show_annot = num_patients <= 30
    show_yticks = num_patients <= 50

    sns.heatmap(
        table.astype(float),
        annot=show_annot,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Spearman 's Rho (ρ)"},
        ax=ax,
        linewidths=0.5 if show_yticks else 0,
        linecolor="whitesmoke",
        yticklabels=show_yticks,
    )
    ax.set_title(
        "Heatmap of Expr/CN Correlation by Patient/Tissue", fontsize=12, pad=12
    )
    ax.set_xlabel("Tissue", fontsize=10)
    ax.set_ylabel("Patient", fontsize=10)

    plt.tight_layout()
    plt.savefig(path.join(output_dir, "corr_heatmap_by_patients.png"), dpi=300)
    plt.close()


def compute_gene_correlations(rna_mat, cnv_mat, gene_id_name_map):
    """ Computes Spearman Correlation between expression and copy numbers for all genes considering all samples. """

    samples_in_expr = set(col for col in rna_mat.columns)
    print("\nSamples in expr table: " + str(len(samples_in_expr)))

    samples_in_cn = set(col for col in cnv_mat.columns)
    print("Samples in cn table: " + str(len(samples_in_cn)))

    valid_samples = sorted(list(samples_in_expr.intersection(samples_in_cn)))
    print("Samples in common: " + str(len(valid_samples)))

    rna_mat = rna_mat.loc[:, valid_samples]
    cnv_mat = cnv_mat.loc[:, valid_samples]

    gene_results = []
    gene_id_name_map_inv = dict(zip(gene_id_name_map.values(), gene_id_name_map.keys()))

    for gene in rna_mat.index:
        rna_vals = rna_mat.loc[gene]
        cnv_vals = cnv_mat.loc[gene]

        # Remove eventual missing values
        mask = rna_vals.notna() & cnv_vals.notna()
        r_clean = rna_vals[mask]
        c_clean = cnv_vals[mask]

        # Verify minimum variance (avoid zero-division in constant series)
        if len(r_clean) >= 3 and r_clean.nunique() > 1 and c_clean.nunique() > 1:
            rho, pval = spearmanr(r_clean, c_clean, nan_policy='omit')
        else:
            rho, pval = np.nan, np.nan

        gene_results.append({
            'gene_id': gene,
            'gene_name': gene_id_name_map_inv.get(gene, 'N/A'),
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
    df_gene_corr.to_csv(path.join(output_dir, "corr_expr_cn_gene_level.tsv"), sep='\t', index=False)


def compute_gene_tissue_correlations(rna_mat, cnv_mat, genes_in_common, df_cn):
    """ Compute correlation between bulk-expr and copy number values for single genes considering each tissue. """
    df_metadata = pd.read_csv(metadata_filepath,                     sep='\t')  # for extracting list of patients and tissues, to visualize what's missing
    df_metadata['sample_id'] = df_metadata['sample_id'].apply(clean_sample_id)

    # dict sample_id -> tissue
    sample_to_tissue = df_metadata.set_index('sample_id')['tissue'].to_dict()

    samples_in_expr = set(col for col in rna_mat.columns)
    print("\nSamples in expr table: " + str(len(samples_in_expr)))

    samples_in_cn = set(col for col in cnv_mat.columns)
    print("Samples in cn table: " + str(len(samples_in_cn)))

    valid_samples = sorted(list(samples_in_expr.intersection(samples_in_cn)))
    print("Samples in common: " + str(len(valid_samples)))

    # samples mapped to tissue in common
    samples_tissue_series = pd.Series(valid_samples).map(sample_to_tissue)
    unique_tissues = [t for t in samples_tissue_series.dropna().unique() if str(t).strip()]

    gene_name_map_inv = dict(zip(df_cn['gene_id'], df_cn['symbol']))
    tissue_corr_dict = {
        gene: {
            'gene_id': gene,
            'gene_name': gene_name_map_inv.get(gene, 'N/A')
        }
        for gene in genes_in_common
    }

    print("Computing correlations for each tissue...")
    for tissue in tqdm(unique_tissues):
        tissue_samples = [s for s in valid_samples if sample_to_tissue.get(s) == tissue]

        # at least 5 samples to consider Spearman relevant
        if len(tissue_samples) >= 5:
            rna_t = rna_mat[tissue_samples]
            cnv_t = cnv_mat[tissue_samples]

            for gene in genes_in_common:
                r_vals = rna_t.loc[gene]
                c_vals = cnv_t.loc[gene]

                # Remove eventual missing values
                mask = r_vals.notna() & c_vals.notna()
                r_clean = r_vals[mask]
                c_clean = c_vals[mask]

                # Verify minimum variance (avoid zero-division in constant series)
                if len(r_clean) >= 3 and r_clean.nunique() > 1 and c_clean.nunique() > 1:
                    rho, pval = spearmanr(r_clean, c_clean, nan_policy='omit')
                else:
                    rho, pval = np.nan, np.nan

                tissue_corr_dict[gene][tissue] = format_correlation(rho, pval)

    df_gene_tissue_corr = pd.DataFrame(list(tissue_corr_dict.values()))

    # columns order: gene_id, gene_name, then tissues
    present_tissues = [t for t in unique_tissues if t in df_gene_tissue_corr.columns]
    cols = ['gene_id', 'gene_name'] + present_tissues
    df_gene_tissue_corr = df_gene_tissue_corr[cols]

    tissue_corr_path = path.join(output_dir, "corr_expr_cn_gene_and_tissue_level.tsv")
    print(f"Saving gene tissue-specific correlation matrix to: {tissue_corr_path}...")
    df_gene_tissue_corr.to_csv(tissue_corr_path, sep='\t', index=False)


    """"" Heatmap for first N genes with highest avg correlation """""

    print("Computing heatmap...")
    if len(present_tissues) > 1:
        plt.figure(figsize=(8, 10))

        df_plot = df_gene_tissue_corr.copy()

        # values with p-value < 0.05 "(ignore)" are set to NaN
        for col in present_tissues:
            df_plot[col] = df_plot[col].astype(str).apply(
                lambda x: np.nan if '(ignore)' in x else x
            )
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

        df_plot['mean_rho'] = df_plot[present_tissues].mean(axis=1)
        top_genes_plot = df_plot.sort_values(by='mean_rho', ascending=False).head(30)

        heatmap_data = top_genes_plot.set_index('gene_name')[present_tissues]

        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    cbar_kws={'label': 'Spearman Rho'})
        plt.title("Top 30 Genes: RNA-CNV Spearman Correlation by Tissue")
        plt.tight_layout()
        plt.savefig(path.join(output_dir, "corr_heatmap_top_genes.png"), dpi=300)
        plt.close()

    return df_gene_tissue_corr


if __name__ == "__main__":
    print("Loading rna seq data...")
    df_rna = pd.read_csv(rna_filepath, sep='\t', index_col=0)
    print("Loading copy numbers data...")
    df_cnv = pd.read_csv(cnv_filepath, sep='\t', dtype={
        'gene_id': 'category',
        'symbol': 'category',
        'sample': 'category',
        'purifiedLogR': 'float32'
    })

    gene_id_to_symbol_map = dict(zip(df_cnv['symbol'].str.upper(), df_cnv['gene_id']))

    #control_genes = pd.Series(['CCNE1', 'ERBB2', 'MYC', 'MECOM', 'MDM2', 'KRAS', 'AKT2']).map(gene_id_to_symbol_map)
    #valid_genes = pd.read_csv(known_drivers_filepath, sep='\t', usecols=["Symbol"])["Symbol"].map(gene_id_to_symbol_map)
    valid_genes = pd.read_csv(selected_genes_filepath, sep='\t')['gene_id']
    #valid_genes = df_rna.index

    rna_sample_map = {col: clean_sample_id(col) for col in df_rna.columns}  # clean samples IDs by keeping only Patient_Tissue
    df_rna = df_rna.rename(columns=rna_sample_map)
    df_rna = df_rna.T.groupby(level=0).aggregate('mean').T  # eventual duplicated cols_order es. D382_iOme1_RNA1 and D382_iOme1_RNA2, aggregated

    df_cnv_filtered = df_cnv[df_cnv['gene_id'].isin(valid_genes)].copy()
    df_cnv_filtered['sample'] = df_cnv_filtered['sample'].apply(clean_sample_id)
    df_cnv_filtered['tot_copies'] = df_cnv_filtered['nMajor'] + df_cnv_filtered['nMinor']

    print("Extracting cn values matrix in correct format...")

    # CNV metric chosen to correlate
    cn_values = 'purifiedLogR'  # purifiedLogR or tot_copies
    df_cnv_pivot = df_cnv_filtered.pivot_table(
        index='gene_id',
        columns='sample',
        values=cn_values,
        aggfunc='mean',
        observed=True,
    )
    common_genes = df_rna.index.intersection(df_cnv_pivot.index)
    print(f"Common Genes in expr and cn tables: {len(common_genes)} / {len(valid_genes)}")

    rna_table = df_rna.loc[common_genes, :]
    cnv_table = df_cnv_pivot.loc[common_genes, :]

    patient_tissue_level_corr(rna_table, cnv_table)
    compute_gene_correlations(rna_table, cnv_table, gene_id_to_symbol_map)
    compute_gene_tissue_correlations(rna_table, cnv_table, common_genes, df_cnv_filtered)

import re
from os import path, makedirs

import numpy as np
import pandas as pd
import pyranges as pr

import config_file_tree
import utils_visualization

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR
cnv_filepath = path.join(preprocessed_dir, "preprocessed_cn.tsv")
rna_filepath = path.join(preprocessed_dir, "preprocessed_bulk_expr.tsv")
metadata_filepath = path.join(preprocessed_dir, "preprocessed_samples_metadata.tsv")

current_dir = path.dirname(path.realpath(__file__))
output_dir = path.join(current_dir, "results")

makedirs(output_dir, exist_ok=True)

def process_cnv_data(
        gnomad_filepath: str = "gnomad.v4.1.sv.sites.bed.gz",
        gencode_filepath: str = "gencode.v36.annotation.gtf",
        min_germline_freq: float = 0.05,
        min_overlap_frac: float = 0.50,) -> pd.DataFrame:
    """ Copy Number normal tissue control.
    Processes CNV somatic data filtering germinal variants (gnomAD/GENCODE). """

    cache_lookup_path = path.join(current_dir, "normal_gene_lookup_cache.tsv")

    if path.exists(cache_lookup_path):
        print("Loading cached normal gene lookup table...")
        normal_gene_lookup = pd.read_csv(cache_lookup_path, sep="\t")
    else:
        cols_to_keep = [
            "#chrom",
            "start",
            "end",
            "svtype",
            "SVLEN",
            "FILTER",
            "AF",
            "AF_nfe",
            "CN_NONREF_FREQ",
        ]

        print("Loading normal cn values file...")
        normal_cnvs = pd.read_csv(
            gnomad_filepath,
            sep="\t",
            header=0,
            usecols=cols_to_keep,
            dtype={
                "#chrom": 'category',
                "svtype": 'category',
                "FILTER": 'category',
            },
        )

        normal_cnvs = normal_cnvs[(normal_cnvs["FILTER"] == "PASS")].copy()  # validity filter

        # dataset considers variations on copy numbers (purifiedLogR), so kept only germlines that alter the number of copies
        cnv_types = ["DEL", "DUP", "MCNV", "CNV"]
        normal_cnvs = normal_cnvs[normal_cnvs["svtype"].isin(cnv_types)].copy()

        # germinal freq. unified: AF if present, else CN_NONREF_FREQ
        normal_cnvs["germline_pop_freq"] = normal_cnvs["AF"].fillna(normal_cnvs["CN_NONREF_FREQ"])

        # la specific freq. unified for European Population (NFE)
        normal_cnvs["germline_nfe_freq"] = normal_cnvs["AF_nfe"].fillna(normal_cnvs["CN_NONREF_FREQ"])

        common_germline_prep = normal_cnvs.rename(
            columns={"#chrom": "Chromosome", "start": "Start", "end": "End"}
        )
        pr_normal = pr.PyRanges(common_germline_prep)

        print("GTF file upload...\n")
        pr_genes = pr.read_gtf(gencode_filepath)
        pr_exons = pr_genes[pr_genes.Feature == "exon"].merge(by="gene_id")

        # Computing total exons length for every gene (sum of exons area)
        df_exons = pr_exons.as_df()
        df_exons["gene_id"] = df_exons["gene_id"].str.split(".").str[0]
        df_exons["exon_len"] = df_exons["End"] - df_exons["Start"]
        gene_total_exons = (
            df_exons.groupby("gene_id")["exon_len"].sum().to_dict()
        )

        print("Mapping and calculating cumulative exon overlaps...\n")
        joined = pr_exons.join(pr_normal, suffix="_normal")
        df_joined = joined.as_df()
        df_joined["gene_id"] = df_joined["gene_id"].str.split(".").str[0]

        # ==================== CALCOLO SOGLIA OVERLAP ====================
        # Calcolo overlap in bp tra variante germinale e singolo esone
        overlap_start = np.maximum(
            df_joined["Start"].values, df_joined["Start_normal"].values
        )
        overlap_end = np.minimum(
            df_joined["End"].values, df_joined["End_normal"].values
        )
        df_joined["overlap_bp"] = np.maximum(0, overlap_end - overlap_start)

        # sum of exon basis covered for each germinal variant and gene
        cov_summary = (
            df_joined.groupby(
                ["gene_id", "Start_normal", "End_normal", "svtype"],
                observed=True
            )
            .agg(
                covered_exon_bp=("overlap_bp", "sum"),
                germline_pop_freq=("germline_pop_freq", "max"),
                germline_nfe_freq=("germline_nfe_freq", "max"),
            )
            .reset_index()
        )

        # 5. Frazione dell'esoma del gene coperto dalla variante germinale
        cov_summary["total_exon_bp"] = cov_summary["gene_id"].map(
            gene_total_exons
        )
        cov_summary["exon_overlap_frac"] = (
                cov_summary["covered_exon_bp"] / cov_summary["total_exon_bp"]
        )

        # 6. Filtro: la CNV germinale deve coprire almeno il 50% dell'esoma complessivo del gene
        valid_germline = cov_summary[
            cov_summary["exon_overlap_frac"] >= min_overlap_frac
            ].copy()
        # ================================================================

        # gene_id aggregation: max germinal freq found
        print("Aggregating genes...\n")
        normal_gene_lookup = (
            valid_germline.groupby("gene_id")
            .agg(
                germline_pop_freq=("germline_pop_freq", "max"),
                germline_nfe_freq=("germline_nfe_freq", "max"),
                has_common_germline_cnv=(
                    "germline_pop_freq",
                    lambda x: (x >= min_germline_freq).any(),  # common CNV (Duplications and Deletions) in normal population (allelic freq. > 1%)
                ),
            )
            .reset_index()
        )

        normal_gene_lookup.to_csv(cache_lookup_path, sep="\t", index=False)
        print(f"Lookup table created and saved to {cache_lookup_path}\n")

    print("Loading and merging cn dataset...\n")
    df_cn_dataset = pd.read_csv(cnv_filepath, sep="\t")

    germline_genes = set(
        normal_gene_lookup.loc[
            normal_gene_lookup["has_common_germline_cnv"], "gene_id"
        ]
    )

    df_cn_dataset["somatic_purifiedLogR"] = df_cn_dataset["purifiedLogR"]
    gene_col = "gene_id"

    # somatic filter (purifiedLogR = 0 if common germinal)
    df_cn_dataset["is_common_germline"] = df_cn_dataset[gene_col].isin(germline_genes)
    #df_cn_dataset.loc[df_cn_dataset["is_common_germline"], "somatic_purifiedLogR"] = 0.0  # kept, will have a different style

    print("CN dataset filtering complete! 'somatic_purifiedLogR' successfully generated.\n")
    print(f"Totale geni con CNV germinale comune individuati: {len(germline_genes)}")  # TODO check con esoni o no
    print(
        f"Righe somatiche contrassegnate come germinali: {df_cn_dataset['is_common_germline'].sum()} su {len(df_cn_dataset)}")
    return  df_cn_dataset


def process_rna_expression(
        gtex_filepath: str = "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct",
        agg_method: str = "median") -> pd.DataFrame:
    """ Expression normal tissue control.
    Processes RNA expression computing the difference between dataset values wrt GTEx normal baseline. """

    print("Loading GTEx RNA expression baseline...\n")
    df_gtex = pd.read_csv(
        gtex_filepath,
        sep="\t",
        skiprows=2,  # skip first 2 rows (format header)
    )
    df_gtex['Name'] = df_gtex['Name'].str.split('.').str[0]

    gtex_tissue_map = {
        'Ova': ['Ovary'],
        'Tub': ['Fallopian_Tube'],
        'Ute': ['Uterus'],
        'Vag': ['Vagina'],
        'Ome': ['Adipose_Visceral_Omentum'],
        'Bow': ['Colon_Sigmoid', 'Colon_Transverse', 'Small_Intestine_Terminal_Ileum'],
        'Adn': ['Ovary', 'Fallopian_Tube'],
        'Per': ['Adipose_Visceral_Omentum'],
        'Mes': ['Adipose_Visceral_Omentum'],
        'LN': ['Spleen', 'Whole_Blood'],
        'Asc': ['Adipose_Visceral_Omentum', 'Fallopian_Tube', 'Ovary'],
        'Plf': ['Adipose_Visceral_Omentum', 'Fallopian_Tube', 'Ovary'],
        'Oth': ['Ovary', 'Fallopian_Tube', 'Uterus', 'Adipose_Visceral_Omentum'],
    }

    print(f"Computing tissue-specific GTEx baselines using '{agg_method}'...")

    # for each tissue mapped
    gtex_baseline_cols = []
    for dataset_tissue, gtex_cols in gtex_tissue_map.items():
        valid_cols = [c for c in gtex_cols if c in df_gtex.columns]
        col_name = f"gtex_baseline_{dataset_tissue}"

        if valid_cols:
            if agg_method == 'median':
                df_gtex[col_name] = df_gtex[valid_cols].median(axis=1)
            else:
                df_gtex[col_name] = df_gtex[valid_cols].mean(axis=1)
        else:
            df_gtex[col_name] = 0.0

        gtex_baseline_cols.append(col_name)

    # fallback (median of all tissues interested)
    all_mapped_gtex_cols = list(set([col for cols in gtex_tissue_map.values() for col in cols if col in df_gtex.columns]))
    df_gtex['gtex_baseline_global'] = df_gtex[all_mapped_gtex_cols].median(axis=1)

    print("Loading tumor RNA expression matrix...")
    df_rna_matrix = pd.read_csv(rna_filepath, sep="\t")
    rna_gene_id_col = df_rna_matrix.columns[0]

    keep_cols = [rna_gene_id_col] + [c for c in df_rna_matrix.columns[1:] if (
            re.search(r"DMSO|RNA1|RNA2", c, re.IGNORECASE) and not re.search(r"Alpelisib", c, re.IGNORECASE)
        )
    ]
    df_rna_matrix = df_rna_matrix[keep_cols]

    # matrix transformed from wide (gene_id x sample_id) to long (gene_id, sample_id, expr_value)
    print("Reshaping RNA matrix to long format...")
    df_rna_long = df_rna_matrix.melt(
        id_vars=[rna_gene_id_col],
        var_name='sample_id',
        value_name='TPM'  # already log-transformed in preprocessing
    )

    # adding 'tissue' col to rna data from metadata
    samples_metadata = pd.read_csv(metadata_filepath, sep="\t", usecols=['sample_id', 'tissue'])
    df_rna_long = df_rna_long.merge(
        samples_metadata,
        on='sample_id',
        how='left'
    )

    """"" Adding baseline values depending on tissue """""

    # 1. baselines GTEx transformed from wide to long (gene_id, tissue, baseline_value)
    df_gtex_long = df_gtex.melt(
        id_vars=['Name'],
        value_vars=gtex_baseline_cols,
        var_name='tissue',
        value_name='gtex_baseline_matched'
    )
    df_gtex_long['tissue'] = df_gtex_long['tissue'].str.replace('gtex_baseline_', '', regex=False)

    # 2. merge of specific baseline value on [gene_id, tissue]
    print("Matching tissue-specific baselines in df_rna_long (vectorized)...")
    df_rna_long = df_rna_long.merge(
        df_gtex_long,
        left_on=[rna_gene_id_col, 'tissue'],
        right_on=['Name', 'tissue'],
        how='left'
    ).drop(columns=['Name'], errors='ignore')

    # 3. merge of global baseline (fallback if tissue not matching)
    df_rna_long = df_rna_long.merge(
        df_gtex[['Name', 'gtex_baseline_global']],
        left_on=rna_gene_id_col,
        right_on='Name',
        how='left'
    ).drop(columns=['Name'], errors='ignore')

    # 4. fallback
    df_rna_long['gtex_baseline_matched'] = (
        df_rna_long['gtex_baseline_matched']
        .fillna(df_rna_long['gtex_baseline_global'])
        .fillna(0.0)
    )
    df_rna_long.drop(columns=['gtex_baseline_global'], inplace=True, errors='ignore')

    print("Computing Delta RNA (Somatic Expression shift)...")
    df_rna_long['TPM'] = df_rna_long['TPM'].fillna(0.0)
    df_rna_long['somatic_delta_RNA'] = df_rna_long['TPM'] - np.log2(df_rna_long['gtex_baseline_matched'] + 1)

    return  df_rna_long


def integrate_cnv_rna(df_cn: pd.DataFrame, df_rna: pd.DataFrame) -> pd.DataFrame:
    """Final merge between CNV and RNA dataframes based on gene and sample."""
    print("Merging CNV dataset and RNA dataset into df_merged...")

    # Estrarre i primi due campi separati da "_" sia per CNV che per RNA
    df_cn["sample"] = df_cn["sample"].str.extract(r"^([^_]+_[^_]+)")[0]
    df_rna["sample"] = df_rna["sample_id"].str.extract(r"^([^_]+_[^_]+)")[0]

    rna_gene_col = df_rna.columns[0]

    rna_cols = [
        rna_gene_col,
        "sample",
        "sample_id",
        "TPM",
        "gtex_baseline_matched",
        "somatic_delta_RNA",
    ]

    merged = df_cn.merge(
        df_rna[rna_cols],
        left_on=["gene_id", "sample"],
        right_on=[rna_gene_col, "sample"],
        how="left",
    )

    print("Integration complete!\n")
    return merged


if __name__ == "__main__":
    # 1. CNV
    df_cn_with_germline = process_cnv_data()

    # 2. RNA
    df_rna_with_baseline = process_rna_expression()

    # 3. Integration
    df_merged = integrate_cnv_rna(df_cn=df_cn_with_germline, df_rna=df_rna_with_baseline)

    # List of interested genes

    ovarian_genes = [
        "PAX8",
        "WT1",
        "FOLR1",
        "MUC16",
        "KLK6",
        "ERBB2",
        "CCNE1",
        "SAMD11",
    ]
    """

    ovarian_genes = [
        "TP53",
        "CSMD3",
        "NF1",
        "BRCA1",
        "PIK3CA",
        "KRAS",
        "CDK12",
        "PTPRB",
        "BRCA2",
    ]"""

    matched_pct = (df_merged['gtex_baseline_matched'] > 0).mean() * 100
    print(f"[INFO] % of genes/tissues correctly associated with GTEx: {matched_pct:.2f}%")

    # Plots
    utils_visualization.plot_cnv_vs_rna_quadrants(
        df_merged=df_merged,
        gene_list=ovarian_genes,
        output_path=path.join(output_dir, "quadrant_plot_ovarian_genes.png"),
    )

    utils_visualization.plot_cnv_rna_dual_heatmap(
        df_merged=df_merged,
        gene_list=ovarian_genes,
        output_path=path.join(output_dir, "dual_heatmap_ovarian_genes.png"),
    )

    # Table extraction of relevant candidates
    high_cn_low_rna, low_cn_high_rna = utils_visualization.extract_cnv_rna_hits(
        df_merged=df_merged, gene_list=ovarian_genes
    )

    print(
        f"Cases 'High CN / Low RNA' found: {len(high_cn_low_rna)}"
    )
    print(
        f"Cases 'Low CN / High RNA' found: {len(low_cn_high_rna)}"
    )

    # Results to tsv
    high_cn_low_rna.to_csv(path.join(output_dir, "hits_high_CN_low_RNA.tsv"), sep="\t", index=False)
    low_cn_high_rna.to_csv(path.join(output_dir, "hits_low_CN_high_RNA.tsv"), sep="\t", index=False)

    print("\nDONE!\n")

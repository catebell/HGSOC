import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_cnv_vs_rna_quadrants(
    df_merged: pd.DataFrame,
    gene_list: list[str],
    cn_threshold: float = 0.3,
    rna_threshold: float = 1.0,
    exclude_common_germline: bool = False,
    output_path: str = None,
):
    """Creates a 4 quadrants Scatter Plot (CNV vs somatic_delta_RNA) for chosen genes list."""

    print("Creating 4 quadrants scatter plot...")
    df_plot = df_merged[df_merged["Gene"].isin(gene_list)].copy()

    if exclude_common_germline and "is_common_germline" in df_plot.columns:
        df_plot = df_plot[~df_plot["is_common_germline"]]

    df_plot = df_plot.dropna(
        subset=["somatic_purifiedLogR", "somatic_delta_RNA"]
    )

    if df_plot.empty:
        print("[ERROR]: no data found for chosen genes.")
        return

    plt.figure(figsize=(11, 7))
    sns.set_theme(style="whitegrid")

    sns.scatterplot(
        data=df_plot,
        x="somatic_purifiedLogR",
        y="somatic_delta_RNA",
        hue="Gene",
        hue_order=gene_list,
        style="is_common_germline",
        #style="tissue" if "tissue" in df_plot.columns else None,
        alpha=0.85,
        s=80,
    )

    # thresholds visualization
    plt.axvline(x=cn_threshold, color="red", linestyle="--", alpha=0.6)
    plt.axvline(x=-cn_threshold, color="grey", linestyle=":", alpha=0.5)
    plt.axhline(y=rna_threshold, color="red", linestyle="--", alpha=0.6)
    plt.axhline(y=-rna_threshold, color="blue", linestyle="--", alpha=0.6)

    plt.title(
        "Multi-Omic integration: CNV vs Delta RNA",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Somatic CNV (somatic_purifiedLogR)", fontsize=11)
    plt.ylabel(
        "Delta Expression vs GTEx (somatic_delta_RNA)", fontsize=11
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"4 quadrants plot saved to: {output_path}")

    plt.close()


def plot_cnv_rna_dual_heatmap(
    df_merged: pd.DataFrame, gene_list: list[str], output_path: str = None
):
    """Creates two joined heatmap (Copy Number and Delta RNA Expression) for selected genes."""

    print("Creating dual heatmap...")
    df_sub = df_merged[df_merged["Gene"].isin(gene_list)].copy()

    if df_sub.empty:
        print("[ERROR]: No data available for heatmap.")
        return

    sample_col = "sample" if "sample" in df_sub.columns else "sample_id"

    # Pivot matrices (Gene x Sample)
    matrix_cn = df_sub.pivot_table(
        index="Gene",
        columns=sample_col,
        values="somatic_purifiedLogR",
        aggfunc="mean",
    )
    matrix_rna = df_sub.pivot_table(
        index="Gene",
        columns=sample_col,
        values="somatic_delta_RNA",
        aggfunc="mean",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.heatmap(
        matrix_cn,
        cmap="coolwarm",
        center=0,
        ax=ax1,
        cbar_kws={"label": "CNV (somatic_purifiedLogR)"},
    )
    ax1.set_title("Somatic Copy Number", fontweight="bold")

    sns.heatmap(
        matrix_rna,
        cmap="PiYG",
        center=0,
        ax=ax2,
        cbar_kws={"label": "Delta RNA (vs GTEx)"},
    )
    ax2.set_title("Difference in RNA Expression", fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Double Heatmap saved to: {output_path}")

    plt.close()


def extract_cnv_rna_hits(
    df_merged: pd.DataFrame,
    gene_list: list[str],
    method: str = "percentile",  # 'percentile', 'std', 'fixed'
    cn_threshold: float = 0.3,
    rna_threshold_high: float = 1.0,
    rna_threshold_low: float = -1.0,
    quantiles_rna: tuple[float, float] = (0.10, 0.90),
    std_cutoff: float = 1.5,
    exclude_common_germline: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts and exports 'hits' for the two cases of interest"""

    print("Extracting cases of interest...")
    df_sub = df_merged[df_merged["Gene"].isin(gene_list)].copy()

    if exclude_common_germline and "is_common_germline" in df_sub.columns:
        df_sub = df_sub[~df_sub["is_common_germline"]]

    if df_sub.empty:
        print("[ERROR] No data remained after filtering.")
        return pd.DataFrame(), pd.DataFrame()

    # Dynamic choice of thresholds for somatic_delta_RNA
    if method == "percentile":
        q_low, q_high = quantiles_rna
        t_low = df_sub["somatic_delta_RNA"].quantile(q_low)
        t_high = df_sub["somatic_delta_RNA"].quantile(q_high)
        print(
            f"Percentile empirical thresholds ({q_low * 100:.0f}° / {q_high * 100:.0f}°): "
            f"Low Delta <= {t_low:.3f}, High Delta >= {t_high:.3f}"
        )

    elif method == "std":
        mean_rna = df_sub["somatic_delta_RNA"].mean()
        std_rna = df_sub["somatic_delta_RNA"].std()
        t_low = mean_rna - std_cutoff * std_rna
        t_high = mean_rna + std_cutoff * std_rna
        print(
            f"Empirical std thresholds (±{std_cutoff}σ): "
            f"Low Delta <= {t_low:.3f}, High Delta >= {t_high:.3f}"
        )

    else:  # 'fixed'
        t_low = rna_threshold_low
        t_high = rna_threshold_high
        print(
            f"Fixed thresholds: Low Delta <= {t_low:.3f}, High Delta >= {t_high:.3f}"
        )

    cols = [
        "Gene",
        "sample" if "sample" in df_sub.columns else "sample_id",
        "tissue",
        "is_common_germline",
        "somatic_purifiedLogR",
        "TPM",
        "gtex_baseline_matched",
        "somatic_delta_RNA",
    ]
    cols = [c for c in cols if c in df_sub.columns]

    # A case: High Copy Number but Low or No Expression
    high_cn_low_rna = df_sub[
        (df_sub["somatic_purifiedLogR"] >= cn_threshold)
        & (df_sub["somatic_delta_RNA"] <= rna_threshold_low)
    ][cols]

    # B case: Normal/Low CN but Expression higher than normal
    low_cn_high_rna = df_sub[
        (df_sub["somatic_purifiedLogR"] <= 0.0)
        & (df_sub["somatic_delta_RNA"] >= rna_threshold_high)
    ][cols]

    return high_cn_low_rna, low_cn_high_rna
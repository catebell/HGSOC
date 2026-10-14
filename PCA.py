from os import path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca_analysis(
    input_data_or_filepath,
    input_metadata_filepath,
    output_dir,
    feature_type="Expr",  # 'Expr' or 'CNV'
    feature_col="cohort",
    cn_metric_label="purifiedLogR",
    scale_features=None,
):
    feature_label = (
        "Expr"
        if feature_type.lower() in ["expr", "expression", "rna"]
        else "CNV"
    )
    output_filepath = path.join(
        output_dir, f"PCA_{feature_label}_{feature_col}_comparison_plots.png"
    )

    print(
        f"\nLoading {feature_label} data for PCA analysis grouped by '{feature_col}'..."
    )
    df_metadata = pd.read_csv(input_metadata_filepath, sep="\t")

    if feature_col not in df_metadata.columns:
        raise ValueError(
            f"Column '{feature_col}' does not exist in {input_metadata_filepath}! "
            f"Available cols: {list(df_metadata.columns)}"
        )

    # LOADING MATRIX (Filepath or DataFrame)
    if isinstance(input_data_or_filepath, str):
        df_data = pd.read_csv(input_data_or_filepath, sep="\t", index_col=0)
    elif isinstance(input_data_or_filepath, pd.DataFrame):
        df_data = input_data_or_filepath.copy()
    else:
        raise TypeError("input_data_or_filepath must be a filepath string or a pandas DataFrame")

    # SAMPLES ALIGN
    sample_ids = [
        c for c in df_data.columns if c in df_metadata["sample_id"].values
    ]
    df_data = df_data[sample_ids]
    df_metadata = (
        df_metadata.set_index("sample_id").loc[sample_ids].reset_index()
    )
    df_metadata[feature_col] = (df_metadata[feature_col].fillna("Unknown").astype(str))

    # Remove genes with more than 20% NaNs and imputes mean
    if df_data.isna().sum().sum() > 0:
        df_data_filtered = df_data.dropna(
            thresh=int(0.8 * df_data.shape[1]), axis=0
        )
        df_data_imputed = df_data_filtered.T.apply(
            lambda col: col.fillna(col.mean()), axis=0
        )
    else:
        df_data_imputed = df_data.T

    X = df_data_imputed.values  # shape: (n_samples, n_genes)

    # NORMALIZATION (CNV StandardScaler)
    if scale_features is None:
        scale_features = feature_label == "CNV"

    if scale_features:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_proc)

    df_pca = pd.DataFrame(
        {
            "sample_id": df_metadata["sample_id"].values,
            "patient_id": df_metadata["patient_id"].values
            if "patient_id" in df_metadata.columns
            else "N/A",
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            feature_col: df_metadata[feature_col].values,
        }
    )

    # SCATTER PLOT PCA
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)

    n_unique = df_pca[feature_col].nunique()
    use_style = feature_col if n_unique <= 6 else None
    palette = "Set1" if n_unique <= 9 else "tab20"

    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue=feature_col,
        style=use_style,
        alpha=0.8,
        s=60,
        palette=palette,
    )
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    plt.title(
        f"{feature_label} PCA by {feature_col.capitalize()}\n(PC1: {var_pc1:.1f}%, PC2: {var_pc2:.1f}%)"
    )
    plt.xlabel(f"PC1 ({var_pc1:.1f}%)")
    plt.ylabel(f"PC2 ({var_pc2:.1f}%)")
    plt.grid(True, linestyle="--", alpha=0.5)

    if n_unique > 6:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # DENSITY PLOT (Avg per Sample)
    plt.subplot(1, 2, 2)
    metric_units = (
        "log2(TPM + 1)" if feature_label == "Expr" else cn_metric_label
    )
    df_metadata["mean_val"] = df_data.mean(axis=0).values

    for group_name, group in df_metadata.groupby(feature_col):
        if len(group) > 1:
            sns.kdeplot(
                group["mean_val"],
                label=f"{group_name}",
                fill=True,
                alpha=0.3,
            )

    plt.title(
        f"Mean {feature_label} ({metric_units}) Distribution by {feature_col.capitalize()}"
    )
    plt.xlabel(f"Mean {metric_units}")
    plt.ylabel("Density")

    if n_unique > 6:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    plt.close()

    # STATISTICAL TEST (ANOVA on PC1)
    valid_groups = [
        group["PC1"].values
        for _, group in df_pca.groupby(feature_col)
        if len(group) >= 2
    ]

    if len(valid_groups) > 1:
        f_stat, p_val = f_oneway(*valid_groups)

        print("\n" + "=" * 60)
        print(
            f"STATISTICAL TEST ON {feature_label.upper()} SEPARATION BY '{feature_col.upper()}' (ANOVA on PC1):"
        )
        print("=" * 60)
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value:     {p_val:.4e}")

        if p_val < 0.05:
            print(
                f"\n-> Result: Statistically significant difference between '{feature_col}' groups along PC1."
            )
        else:
            print(
                f"\n-> Result: No statistically significant difference detected along PC1 for feature '{feature_col}'."
            )
    else:
        print(
            f"\n-> ANOVA not executable: requires >= 2 groups with >= 2 samples for feature '{feature_col}'."
        )

    return df_pca

from os import path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.decomposition import PCA


def run_pca_analysis(input_bulk_filepath, input_metadata_filepath, output_dir, feature_col="cohort"):
    output_filepath = path.join(output_dir, f"PCA_{feature_col}_comparison_plots.png")

    print(f"\nLoading data for PCA analysis by {feature_col}...")
    df_metadata = pd.read_csv(input_metadata_filepath, sep="\t")

    if feature_col not in df_metadata.columns:
        raise ValueError(
            f"Column '{feature_col}' not exists in {input_metadata_filepath}! "
            f"Available cols: {list(df_metadata.columns)}"
        )

    df_bulk = pd.read_csv(input_bulk_filepath, sep="\t", index_col=0)

    # samples alignment
    sample_ids = [c for c in df_bulk.columns if c in df_metadata['sample_id'].values]
    df_bulk = df_bulk[sample_ids]
    df_metadata = df_metadata.set_index('sample_id').loc[sample_ids].reset_index()
    df_metadata[feature_col] = df_metadata[feature_col].fillna('Unknown').astype(str)

    print("Computing PCA scatter plot...")
    X = df_bulk.T.values  # shape: (n_samples, n_genes)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        feature_col: df_metadata[feature_col].values,
    })

    # Plot PCA
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)

    n_unique = df_pca[feature_col].nunique()

    use_style = feature_col if n_unique <= 6 else None
    palette = 'Set1' if n_unique <= 9 else 'tab20'

    sns.scatterplot(
        data=df_pca, x='PC1', y='PC2', hue=feature_col, style=feature_col,
        alpha=0.8, s=60, palette=palette
    )
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    plt.title(f"PCA by {feature_col.capitalize()}\n(PC1: {var_pc1:.1f}%, PC2: {var_pc2:.1f}%)")
    plt.xlabel(f"PC1 ({var_pc1:.1f}%)")
    plt.ylabel(f"PC2 ({var_pc2:.1f}%)")
    plt.grid(True, linestyle='--', alpha=0.5)

    if n_unique > 6:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # ----------------------------------------------------
    # 2. DENSITY PLOT (Avg Expression for Sample)
    # ----------------------------------------------------
    plt.subplot(1, 2, 2)
    df_metadata['mean_expression'] = df_bulk.mean(axis=0).values

    for group_name, group in df_metadata.groupby(feature_col):
        if len(group) > 1:
            sns.kdeplot(
                group['mean_expression'],
                label=f"{group_name}",
                fill=True, alpha=0.3
            )

    plt.title(f"Mean Expr. Distribution for Sample by {feature_col.capitalize()}")
    plt.xlabel("Mean Expr.log2(TPM + 1)")
    plt.ylabel("Density")

    if n_unique > 6:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend()

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # 3. STATISTICAL TEST (ANOVA on PC1)
    # ----------------------------------------------------
    valid_groups = [group['PC1'].values for _, group in df_pca.groupby(feature_col) if len(group) >= 2]  # if groups > 2

    if len(valid_groups) > 1:
        f_stat, p_val = f_oneway(*valid_groups)

        print("\n" + "=" * 60)
        print(f"STATISTICAL TEST ON SEPARATION BY '{feature_col.upper()}' (ANOVA on PC1):")
        print("=" * 60)
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value:     {p_val:.4e}")

        if p_val < 0.05:
            print(
                f"\n-> Result: Exists a statistically significant difference between '{feature_col}' groups along PC1.")
        else:
            print(
                f"\n-> Result: No statistically significant difference detected along PC1 for feature '{feature_col}'.")
    else:
        print(
            f"\n-> ANOVA not executable: needs at least 2 groups with >= 2 samples for feature '{feature_col}'.")
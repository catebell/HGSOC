from os import path, makedirs
from shutil import rmtree

import torch
from numpy import column_stack
from torch import tensor, save
from torch_geometric.data import Data
from pandas import read_csv, DataFrame
from tqdm import tqdm

import config_file_tree
from ppi_build import build_edge_index

""""" Data conversion in PyTorch Geometric Data objects that will be passed to the GNN.
Each graph (Data) will represent a single tissue (its genes = nodes, ppi = edges, expression values = node features) """""


""""" paths and parameters """""

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR
preprocessed_expr_matrix_filepath = path.join(preprocessed_dir, 'preprocessed_bulk_expr.tsv')
preprocessed_cn_filepath = path.join(preprocessed_dir, 'preprocessed_cn.tsv')
preprocessed_metadata_filepath = path.join(preprocessed_dir, 'preprocessed_samples_metadata.tsv')

selected_genes_filepath = path.join(preprocessed_dir, "selected_genes.tsv")

output_dir = config_file_tree.PROCESSED_GRAPHS_DIR
if path.exists(output_dir):
    rmtree(output_dir)
makedirs(output_dir, exist_ok=True)

output_metadata_filepath = path.join(output_dir, "associated_metadata.tsv")


TASK = config_file_tree.TASK
TARGET_COL = config_file_tree.TARGET_COL

if TASK == 'survival':
    TIME_COL = config_file_tree.TIME_COL


def cn_to_rna_mapping(rna_df, cn_df):
    rna_sample_ids = rna_df.columns
    cn_samples_ids = cn_df.columns

    # map: CN sample_id -> tissue (es. 'H092_r1Asc1_DNA1' -> 'H092_r1Asc1')
    cn_sample_map = {cn_s: str(cn_s).split('_DNA')[0] for cn_s in cn_samples_ids}

    rna_to_cn_map = {}
    mapping_rows = []
    for rna_s in rna_sample_ids:
        # find CN samples with prefix = prefix in rna samples
        matches = [
            cn_s for cn_s, tissue_prefix in cn_sample_map.items()
            if rna_s.startswith(tissue_prefix + "_") or rna_s == tissue_prefix
        ]

        if len(matches) == 1:
            rna_to_cn_map[rna_s] = matches[0]
            mapping_rows.append({'rna_sample': rna_s, 'cn_sample': matches[0], 'status': 'matched'})
        elif len(matches) > 1:
            mapping_rows.append({'rna_sample': rna_s, 'cn_sample': None, 'status': f'ambiguous ({matches})'})  # seems 0
        else:
            mapping_rows.append({'rna_sample': rna_s, 'cn_sample': None, 'status': 'unmapped'})

    df_map = DataFrame(mapping_rows)

    print("\n--- SAMPLES MAPPING RECAP ---")
    print(df_map['status'].value_counts())
    print("\n")

    # CN samples mapped to many RNA samples (1-a-N)
    matched_df = df_map[df_map['status'] == 'matched']
    cn_counts = matched_df['cn_sample'].value_counts()
    multi_mapped_cn = cn_counts[cn_counts > 1]

    print(f"--- CN SAMPLES MAPPED TO MULTIPLE RNA SAMPLES: {len(multi_mapped_cn)} ---")
    for cn_s, count in multi_mapped_cn.items():
        associated_rna = matched_df[matched_df['cn_sample'] == cn_s]['rna_sample'].tolist()
        print(f"\nCN sample: '{cn_s}' -> Mapped to {count} RNA samples:")
        for rna in associated_rna:
            print(f"   - {rna}")

    unmapped_rna = df_map[df_map['status'] == 'unmapped']['rna_sample'].tolist()
    if unmapped_rna:
        print(f"\n--- UNMATCHED: {len(unmapped_rna)} RNA samples with no CN match ---")
        print(unmapped_rna[:10], "... (first 10)" if len(unmapped_rna) > 10 else "")

    return rna_to_cn_map, df_map


def generate_all_graphs():
    print("Extracting edge index...")
    edge_index, gene_to_idx = build_edge_index()

    genes_ordered = sorted(gene_to_idx, key=gene_to_idx.get)  # sort to 0,1,2,...
    print(f"Number of nodes (genes) in graph: {len(genes_ordered)}\n")

    print("Loading RNA expression matrix...")
    df_expr = read_csv(preprocessed_expr_matrix_filepath, sep="\t", index_col=0)
    print(f"    Original expression matrix shape: {df_expr.shape} (Genes x Samples)")

    print("Loading CN values matrix...")
    df_cn = read_csv(preprocessed_cn_filepath, sep='\t', dtype={
        'gene_id': 'category',
        'symbol': 'category',
        'sample': 'category',
        'purifiedLogR': 'float32'
    })
    df_cnv_pivot = df_cn.pivot_table(
        index='gene_id',
        columns='sample',
        values='purifiedLogR',
        aggfunc='mean',
        observed=True,
    )
    print(f"    CN Pivot matrix shape: {df_cnv_pivot.shape} (Genes x Samples)")

    rna_to_cn_map, df_matches = cn_to_rna_mapping(df_expr, df_cnv_pivot)
    print(f"\n[ALIGNMENT] Valid samples matched (RNA <-> CNV): {len(rna_to_cn_map)}")

    """"" aligning genes and samples between tables """""
    matched_rna_samples = list(rna_to_cn_map.keys())
    matched_cn_samples = list(rna_to_cn_map.values())

    df_expr_aligned = df_expr[matched_rna_samples].reindex(genes_ordered).fillna(0.0)  # missing nodes/genes filled with 0 expression
    print(f"    Aligned matrix shape: {df_expr_aligned.shape}\n")

    df_cnv_aligned = df_cnv_pivot[matched_cn_samples].copy()
    df_cnv_aligned.columns = matched_rna_samples  # renamed with same RNA samples_ids
    df_cnv_aligned = df_cnv_aligned.reindex(index=genes_ordered).fillna(0.0)
    print(f"    Aligned CN matrix shape: {df_cnv_aligned.shape}\n")

    print("Generating PyTorch Geometric graph objects...")
    for sample_id in tqdm(matched_rna_samples):
        expr_vals = df_expr_aligned[sample_id].values
        cn_vals = df_cnv_aligned[sample_id].values

        # concatenating feature vectors for single sample -> shape: [Num_Genes, 2]
        node_features = column_stack([expr_vals, cn_vals])
        x = tensor(node_features, dtype=torch.float32)

        graph = Data(
            x=x,
            edge_index=edge_index,
            sample_id=sample_id
        )

        save_path = path.join(output_dir, f"{sample_id}.pt")
        save(graph, save_path)

    print(f"Save completed for {len(matched_rna_samples)} graphs in {output_dir}\n")
    print(f"Node feature tensor shape (x): [{len(genes_ordered)}, 2]\n")

    metadata =read_csv(preprocessed_metadata_filepath, sep='\t', index_col=0)
    metadata = metadata.loc[matched_rna_samples,:]
    metadata.to_csv(output_metadata_filepath, sep='\t', index=True)
    print(f"Associated metadata saved to: {output_metadata_filepath}")

if __name__ == "__main__":
    generate_all_graphs()

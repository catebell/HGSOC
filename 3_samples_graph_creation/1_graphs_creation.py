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

output_dir = config_file_tree.PROCESSED_GRAPHS_DIR
if path.exists(output_dir):
    rmtree(output_dir)
makedirs(output_dir, exist_ok=True)

output_metadata_filepath = path.join(output_dir, "associated_metadata.tsv")


def generate_all_graphs():
    print("Extracting edge index...")
    edge_index, gene_to_idx = build_edge_index()

    genes_ordered = sorted(gene_to_idx, key=gene_to_idx.get)  # sort to 0,1,2,...
    print(f"Number of nodes (genes) in graph: {len(genes_ordered)}\n")

    print("Loading preprocessed RNA expression matrix...")
    df_expr = read_csv(preprocessed_expr_matrix_filepath, sep="\t", index_col=0)
    print(f"    Original expression matrix shape: {df_expr.shape} (Genes x Samples)")

    print("Loading preprocessed CN values matrix...")
    df_cn = read_csv(preprocessed_cn_filepath, sep='\t', dtype={
        'gene_id': 'category',
        'symbol': 'category',
        'sample_id': 'category',
        'purifiedLogR': 'float32'
    })
    df_cnv_pivot = df_cn.pivot_table(
        index='gene_id',
        columns='sample_id',
        values='purifiedLogR',
        aggfunc='mean',
        observed=True,
    )
    print(f"    CN Pivot matrix shape: {df_cnv_pivot.shape} (Genes x Samples)")

    """"" aligning genes and samples between tables """""
    matched_samples = [s for s in df_expr.columns if s in df_cnv_pivot.columns]
    print(f"\n[ALIGNMENT] Valid samples matched (RNA <-> CNV): {len(matched_samples)}")

    df_expr_aligned = df_expr[matched_samples].reindex(genes_ordered).fillna(0.0)  # missing nodes/genes filled with 0 expression
    print(f"    Aligned matrix shape: {df_expr_aligned.shape}\n")

    df_cnv_aligned = df_cnv_pivot[matched_samples].reindex(index=genes_ordered).fillna(0.0)
    print(f"    Aligned CN matrix shape: {df_cnv_aligned.shape}\n")

    print("Generating PyTorch Geometric graph objects...")
    for sample_id in tqdm(matched_samples):
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

    print(f"Save completed for {len(matched_samples)} graphs in {output_dir}\n")
    print(f"Node feature tensor shape (x): [{len(genes_ordered)}, 2]\n")

    metadata =read_csv(preprocessed_metadata_filepath, sep='\t', index_col=0)
    metadata = metadata.loc[matched_samples,:]
    metadata.to_csv(output_metadata_filepath, sep='\t', index=True)
    print(f"Associated metadata saved to: {output_metadata_filepath}")


if __name__ == "__main__":
    generate_all_graphs()

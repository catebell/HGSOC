from os import path, makedirs
from shutil import rmtree

from torch import tensor, float, save
from torch_geometric.data import Data
from pandas import read_csv
from tqdm import tqdm

import config_file_tree
from ppi_build import build_edge_index

""""" Data conversion in PyTorch Geometric Data objects that will be passed to the GNN.
Each graph (Data) will represent a single tissue (its genes = nodes, ppi = edges, expression values = node features) """""


""""" paths and parameters """""

preprocessed_dir = config_file_tree.PREPROCESSED_DATA_DIR
preprocessed_expr_matrix_filepath = path.join(preprocessed_dir, 'bulk_expr_top_genes_log2.tsv')

output_dir = config_file_tree.PROCESSED_GRAPHS_DIR
if path.exists(output_dir):
    rmtree(output_dir)
makedirs(output_dir, exist_ok=True)

TASK = config_file_tree.TASK
TARGET_COL = config_file_tree.TARGET_COL

if TASK == 'survival':
    TIME_COL = config_file_tree.TIME_COL


def generate_all_graphs():
    print("Extracting edge index...")
    edge_index, gene_to_idx = build_edge_index()

    genes_ordered = sorted(gene_to_idx, key=gene_to_idx.get)  # sort to 0,1,2,...
    print(f"    Number of nodes (genes) in graph: {len(genes_ordered)}")

    print("Loading RNA expression matrix...")
    df_expr = read_csv(preprocessed_expr_matrix_filepath, sep="\t", index_col=0)
    print(f"    Original expression matrix shape: {df_expr.shape} (Genes x Samples)")

    df_expr_aligned = df_expr.reindex(genes_ordered).fillna(0)  # missing nodes/genes filled with 0 expression
    sample_ids = df_expr_aligned.columns.tolist()
    print(f"    Aligned matrix shape: {df_expr_aligned.shape}")

    for sample_id in tqdm(sample_ids):
        # feature vector for single sample -> shape: [Num_Genes, 1]
        sample_values = df_expr_aligned[sample_id].values
        x = tensor(sample_values, dtype=float).unsqueeze(1)

        graph = Data(
            x=x,
            edge_index=edge_index,
            sample_id=sample_id
        )

        save_path = path.join(output_dir, f"{sample_id}.pt")
        save(graph, save_path)

    print(f"Save completed for {len(sample_ids)} graphs in {output_dir}\n")

if __name__ == "__main__":
    generate_all_graphs()

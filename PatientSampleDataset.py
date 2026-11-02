import os

import torch
from torch_geometric.data import Dataset
from torch_geometric.transforms import VirtualNode

import config_file_tree


def update_graphs_metadata(graphs_list, metadata_processed_df, feature_cols=None):
    """ To upload/update metadata after feature processing. feature_cols must contain only the processed features (no target, no ids). """

    if not feature_cols:
        for graph in graphs_list:
            graph.metadata_x = torch.empty((1, 0), dtype=torch.float32)
        return

    for graph in graphs_list:
        if graph.sample_id in metadata_processed_df.index:
            feats = metadata_processed_df.loc[graph.sample_id, feature_cols].values.astype('float32')
            graph.metadata_x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # Shape: [1, N_features]
        else:
            graph.metadata_x = torch.empty((1, 0), dtype=torch.float32)


class PatientSampleDataset(Dataset):
    def __init__(self, graphs_dir, samples_metadata_df, task, target_col=None, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)

        # preventing duplicates (should not be any)
        self.sample_metadata_df = samples_metadata_df.drop_duplicates(subset=['sample_id']).set_index('sample_id')
        self.feature_cols = []  # filled later on update
        self.target_col = target_col
        self.task = task
        self.graphs_dir = graphs_dir

        # keeping self.sample_metadata_df order and checking .pt existence
        self.graph_filenames = []
        for sample_id in self.sample_metadata_df.index:
            f_name = f"{sample_id}.pt"
            if os.path.exists(os.path.join(graphs_dir, f_name)):
                self.graph_filenames.append(f_name)

        # Parsing node feature configuration for ablation/selection
        node_feature_config = getattr(config_file_tree, "NODE_FEATURE_INDEX", None)
        all_node_feature_names = getattr(config_file_tree, "NODE_FEATURE_NAMES", [])

        if node_feature_config is None:
            self.selected_indices = list(range(len(all_node_feature_names))) if all_node_feature_names else None
        elif isinstance(node_feature_config, int):
            self.selected_indices = [node_feature_config]  # 2D tensor [num_nodi, 1]
        elif isinstance(node_feature_config, (list, tuple)):
            self.selected_indices = list(node_feature_config)
        else:
            raise ValueError(f"Invalid NODE_FEATURE_INDEX format: {type(node_feature_config)}")

        if self.selected_indices is not None and all_node_feature_names:
            self.selected_node_feature_names = [all_node_feature_names[i] for i in self.selected_indices]
        else:
            self.selected_node_feature_names = all_node_feature_names


    def len(self):
        return len(self.graph_filenames)


    def get(self, idx):
        f_name = self.graph_filenames[idx]
        file_path = os.path.join(self.graphs_dir, f_name)
        data = torch.load(file_path, weights_only=False)

        sample_id = os.path.splitext(f_name)[0]
        data.sample_id = sample_id
        data.patient_id = str(self.sample_metadata_df.loc[sample_id, 'patient_id'])

        # node features dynamic slicing
        if self.selected_indices is not None:
            data.x = data.x[:, self.selected_indices]
        data.node_feature_names = self.selected_node_feature_names

        # sample target association
        if self.task == 'classification':
            y_val = self.sample_metadata_df.loc[sample_id, self.target_col]
            data.y = torch.tensor([y_val], dtype=torch.long)
        elif self.task == 'survival':
            target_time = self.sample_metadata_df.loc[sample_id, 'overall_survival_days']
            target_event = self.sample_metadata_df.loc[sample_id, 'deceased']
            data.y_time = torch.tensor([target_time], dtype=torch.float)
            data.y_event = torch.tensor([target_event], dtype=torch.float)

        # base metadata_x initialization (overwritted later after processing)
        data.metadata_x = None

        return data



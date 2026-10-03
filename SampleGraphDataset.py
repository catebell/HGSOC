import os

import pandas as pd
import torch
from torch_geometric.data import Dataset

# metadata columns about to be excluded from features
id_and_string_cols = ['sample_id', 'patient_id', 'cohort', 'phase', 'extraction',]

leaking_cols = []

class PatientTissueDataset(Dataset):
    def __init__(self, graphs_dir, samples_metadata_df, task, target_col=None, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)

        # preventing duplicates (should not be any)
        self.sample_metadata_df = samples_metadata_df.drop_duplicates(subset=['sample_id']).set_index('sample_id')
        self.target_col = target_col
        self.task = task
        self.graphs_dir = graphs_dir

        # filtering to load only graphs of samples actually present in the metadata file, to prevent eventual errors
        valid_sample_ids = set(self.sample_metadata_df.index)
        raw_filenames = [f for f in os.listdir(graphs_dir) if f.endswith('.pt')]
        self.graph_filenames = []

        for f_name in raw_filenames:
            # ids extracted from filenames ('ID.pt' -> 'ID')
            sample_id = os.path.splitext(f_name)[0]
            if sample_id in valid_sample_ids:
                self.graph_filenames.append(f_name)


        """"" samples metadata columns setup """""

        self.sample_feature_cols = ['phase_label', 'tissue', 'is_fluid']


        if task == 'classification':
            self.sample_feature_cols = [c for c in self.sample_metadata_df.columns if c not in [target_col]
                                        and c not in leaking_cols
                                        #and c not in id_and_string_cols
                                        and pd.api.types.is_numeric_dtype(self.sample_metadata_df[c])]

        elif task == 'survival':
            self.sample_feature_cols = [c for c in self.sample_metadata_df.columns if c not in ['overall_survival_days', 'deceased']
                                        and c not in leaking_cols
                                        and not c.startswith('new_neoplasm_event_type_')
                                        and not c.startswith('progression_determined_by_')
                                        and not c.startswith('person_neoplasm_cancer_status_')]


    def len(self):
        return len(self.graph_filenames)

    def get(self, idx):
        file_path = os.path.join(self.graphs_dir, self.graph_filenames[idx])
        data = torch.load(file_path, weights_only=False)

        data.sample_id = os.path.splitext(self.graph_filenames[idx])[0]
        data.patient_id = str(self.sample_metadata_df.loc[data.sample_id, 'patient_id'])

        # label (y) association with patient
        if self.task == 'classification':
            y_val = self.sample_metadata_df.loc[data.sample_id, self.target_col]
            data.y = torch.tensor([y_val], dtype=torch.long)  # torch.float if regression
        elif self.task == 'survival':
            target_time = self.sample_metadata_df.loc[data.sample_id, 'overall_survival_days']
            target_event = self.sample_metadata_df.loc[data.sample_id, 'deceased']

            data.y_time = torch.tensor([target_time], dtype=torch.float)
            data.y_event = torch.tensor([target_event], dtype=torch.float)

        # sample features association
        sample_metadata_features = self.sample_metadata_df.loc[data.sample_id, self.sample_feature_cols].values

        # Eventual NaNs imputation  # TODO check ma non dovrebbero essercene
        sample_metadata_features = pd.to_numeric(sample_metadata_features, errors='coerce')
        sample_metadata_features = [0.0 if pd.isna(v) else float(v) for v in sample_metadata_features]

        data.metadata_x = torch.tensor(sample_metadata_features, dtype=torch.float).unsqueeze(0)  # Shape [1, Num_Metadata_Feats]

        return data


    def update_metadata_df(self, new_metadata_df, new_feature_cols=None):
        self.sample_metadata_df = new_metadata_df.drop_duplicates(subset=['sample_id']).set_index(
            'sample_id')  # preventing duplicates

        if new_feature_cols is not None:
            self.sample_feature_cols = new_feature_cols

        valid_sample_ids = set(self.sample_metadata_df.index)
        self.graph_filenames = [
            f for f in self.graph_filenames
            if os.path.splitext(f)[0] in valid_sample_ids
        ]
import os

import pandas as pd
import torch
from torch_geometric.data import Dataset

# columns about future to be excluded because they give away the prediction (when target class is one of them)
id_and_string_cols = [
    'patient_id', 'phase', 'phase_label', 'tissue', 'tissue_group',
    'subsample', 'sample_class', 'extraction', 'disease_code', 'bcr_patient_barcode'
]

leaking_cols = []

class PatientTissueDataset(Dataset):
    def __init__(self, graphs_dir, clinical_df, task, target_col=None, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)

        # preventing duplicates (should not be any)
        self.clinical_df = clinical_df.drop_duplicates(subset=['sample_id']).set_index('sample_id')
        self.target_col = target_col
        self.task = task
        self.graphs_dir = graphs_dir

        # filtering to load only graphs of samples actually present in the clinical file, to prevent eventual errors
        valid_sample_ids = set(self.clinical_df.index)
        raw_filenames = [f for f in os.listdir(graphs_dir) if f.endswith('.pt')]
        self.graph_filenames = []

        for f_name in raw_filenames:
            # ids extracted from filenames ('ID.pt' -> 'ID')
            sample_id = os.path.splitext(f_name)[0]
            if sample_id in valid_sample_ids:
                self.graph_filenames.append(f_name)


        """"" clinical columns setup """""

        self.clinical_feature_cols = ['tissue', 'tissue_group_Asc', 'tissue_group_Ome', 'tissue_group_Ova', 'tissue_group_Per']
        '''
        if task == 'classification':
            self.clinical_feature_cols = [c for c in self.clinical_df.columns if c not in [target_col]
                                          and c not in leaking_cols
                                          and c not in id_and_string_cols
                                          and pd.api.types.is_numeric_dtype(self.clinical_df[c])]
        '''
        '''
        elif task == 'survival':
            self.clinical_feature_cols = [c for c in self.clinical_df.columns if c not in ['overall_survival_days', 'deceased']
                                          and c not in leaking_cols
                                          and not c.startswith('new_neoplasm_event_type_')
                                          and not c.startswith('progression_determined_by_')
                                          and not c.startswith('person_neoplasm_cancer_status_')]
        '''

    def len(self):
        return len(self.graph_filenames)

    def get(self, idx):
        file_path = os.path.join(self.graphs_dir, self.graph_filenames[idx])
        data = torch.load(file_path, weights_only=False)

        data.sample_id = os.path.splitext(self.graph_filenames[idx])[0]
        data.patient_id = str(self.clinical_df.loc[data.sample_id, 'patient_id'])

        # label (y) association with patient
        if self.task == 'classification':
            y_val = self.clinical_df.loc[data.sample_id, self.target_col]
            data.y = torch.tensor([y_val], dtype=torch.long)  # torch.float if regression
        elif self.task == 'survival':
            target_time = self.clinical_df.loc[data.sample_id, 'overall_survival_days']
            target_event = self.clinical_df.loc[data.sample_id, 'deceased']

            data.y_time = torch.tensor([target_time], dtype=torch.float)
            data.y_event = torch.tensor([target_event], dtype=torch.float)

        # clinical features association
        clin_feats = self.clinical_df.loc[data.sample_id, self.clinical_feature_cols].values

        # Eventual NaNs imputation  # TODO should not be
        clin_feats = pd.to_numeric(clin_feats, errors='coerce')
        clin_feats = [0.0 if pd.isna(v) else float(v) for v in clin_feats]

        data.clinical_x = torch.tensor(clin_feats, dtype=torch.float).unsqueeze(0)  # Shape [1, Num_Clin_Feats]

        return data


    def update_clinical_df(self, new_df_clinical):
        self.clinical_df = new_df_clinical.drop_duplicates(subset=['sample_id']).set_index(
            'sample_id')  # preventing duplicates
        valid_sample_ids = set(self.clinical_df.index)
        self.graph_filenames = [
            f for f in self.graph_filenames
            if os.path.splitext(f)[0] in valid_sample_ids
        ]
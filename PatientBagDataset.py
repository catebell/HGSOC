import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class PatientBagDataset(Dataset):

  def __init__(self, pyg_samples_list, patient_target_map, patient_clinical_map=None):
    """
    - pyg_samples_list: PyG graphs list (es. train_val_dataset subset)
    - patient_target_map: dict {patient_id: label_outcome}
    - patient_clinical_map: dict {patient_id: clinical_tensor_features}
    """
    self.patient_bags = {}

    # graphs grouped by patient
    for graph in pyg_samples_list:
      patient_id = graph.patient_id

      # if patient retrieved from samples doesn't have a valid clinical record, ignore for MIL
      if patient_id is None or patient_id not in patient_target_map:
        continue

      # if patient missing from clinical map, skipped
      if patient_clinical_map is not None and patient_id not in patient_clinical_map:
        continue

      if patient_id not in self.patient_bags:  # if patient not already seen
        self.patient_bags[patient_id] = []
      self.patient_bags[patient_id].append(graph)

    self.patient_ids = list(self.patient_bags.keys())
    self.patient_target_map = patient_target_map
    self.patient_clinical_map = patient_clinical_map

  def __len__(self):
    return len(self.patient_ids)

  def __getitem__(self, index):
    patient_id = self.patient_ids[index]
    samples_graphs = self.patient_bags[patient_id]

    # Crea il batch di tessuti per il singolo paziente
    patient_bag = Batch.from_data_list(samples_graphs)

    # Salviamo il numero di tessuti e il target come tensori 1D per consentire a PyG di concatenarli
    patient_bag.num_samples = torch.tensor([len(samples_graphs)], dtype=torch.long)
    patient_bag.patient_y = torch.tensor([self.patient_target_map[patient_id]], dtype=torch.long)
    patient_bag.patient_id = [patient_id]

    if self.patient_clinical_map is not None and patient_id in self.patient_clinical_map:
      patient_bag.clinical_x = self.patient_clinical_map[patient_id].unsqueeze(0)  # Shape: [1, num_clinical_features]

    return patient_bag
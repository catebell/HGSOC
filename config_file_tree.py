from os import path


DATASET_DIR = path.join("..", "1_dataset", "data_HERCULES_DECIDER")
RNA_DATA_DIR = path.join(DATASET_DIR, "bulk_RNA")
RNA_BULK_RAW_FILE = path.join(RNA_DATA_DIR, "bulk_expr.tsv")

RNA_EXTRACTED_FILE = path.join(DATASET_DIR, "extracted", "bulk_expr_cohort_H.tsv")  # in this case-project, cohort to use



""""" training parameters """""

TASK = "classification"
#TASK = 'survival'

TARGET_COL = 'tissue'

SURVIVAL_EVENT_COL = 'deceased'
SURVIVAL_TIME_COL = 'overall_survival_days'

#USE_CLINICAL_DATA = True  # available (and filtered) metadata included or not in embeddings creation
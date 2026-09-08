from os import path

DATASET_NAME = "data_HERCULES_DECIDER"

DATASET_DIR = path.join("..", "1_dataset", DATASET_NAME)
RNA_DATA_DIR = path.join(DATASET_DIR, "bulk_RNA")
RNA_BULK_RAW_FILE = path.join(RNA_DATA_DIR, "bulk_expr.tsv")
DECONVOLUTION_W_FILE = path.join(RNA_DATA_DIR, "W.tsv")

EXTRACTED_DATA_DIR = path.join("..", "1_dataset", "extracted_data", DATASET_NAME)

PREPROCESSED_DATA_DIR = path.join("..", "2_preprocessing", "preprocessed_data", DATASET_NAME)


""""" training parameters """""

TASK = "classification"
#TASK = 'survival'

TARGET_COL = 'tissue'

SURVIVAL_EVENT_COL = 'deceased'
SURVIVAL_TIME_COL = 'overall_survival_days'

USE_CLINICAL_DATA = True  # available (and filtered) metadata included or not in embeddings creation
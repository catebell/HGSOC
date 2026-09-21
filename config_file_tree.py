from os import path

DATASET_NAME = "data_HERCULES_DECIDER"

DATASET_DIR = path.join("..", "1_dataset", DATASET_NAME)
RNA_DATA_DIR = path.join(DATASET_DIR, "bulk_RNA")
RNA_BULK_RAW_FILE = path.join(RNA_DATA_DIR, "bulk_expr.tsv")
DECONVOLUTION_W_FILE = path.join(RNA_DATA_DIR, "W.tsv")
CLINICAL_DATA_FILE = path.join(DATASET_DIR, "Clinical export 300123_Decider_Collab.xlsx")  # TODO cambiare con nuovo e adattare

EXTRACTED_DATA_DIR = path.join("..", "1_dataset", "extracted_raw_data", DATASET_NAME)

PREPROCESSED_DATA_DIR = path.join("..", "2_preprocessing", "preprocessed_data", DATASET_NAME)

PROCESSED_GRAPHS_DIR = path.join("..", "4_samples_graph_creation", "processed_graphs", DATASET_NAME)


""""" training parameters """""

RUN_PCA = True

USE_SAMPLES_METADATA = False  # available (and filtered) metadata included or not in embeddings creation
USE_CLINICAL_DATA = False  # available (and filtered) patients data included or not in embeddings creation

TASK = "classification"
#TASK = 'survival'

if TASK == 'classification':
    TARGET_COL = 'Primary_therapy_outcome' # 'phase_label'

    # PFI
    # TARGET_COL =

elif TASK == 'survival':
    # OS - Overall Survival
    #TARGET_COL = 'Survival'  # SURVIVAL_EVENT_COL
    #TIME_COL = 'OS_days'  # SURVIVAL_TIME_COL

    # PFS - Progression-Free Survival
    TARGET_COL = 'Progression_Yes_No'  # PROGRESSION_EVENT_COL
    TIME_COL = 'PFS_days'  # PROGRESSION_TIME_COL
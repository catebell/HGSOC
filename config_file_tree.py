from os import path

ROOT_DIR = path.dirname(path.abspath(__file__))
DATASET_NAME = "data_HERCULES_DECIDER"

DATASET_DIR = path.join(ROOT_DIR, "1_dataset", DATASET_NAME)
ORIGINAL_RNA_DIR = path.join(DATASET_DIR, "expression_tables_updated_pipeline")
ORIGINAL_ASCAT_CN_FILE = path.join(DATASET_DIR, "ASCAT_combinedCnGenes_23Jun2025.tsv")
ORIGINAL_PURPLE_CN_FILE = path.join(DATASET_DIR, "segmentation_info_final.tsv")
DECONVOLUTION_W_FILE = path.join(DATASET_DIR, "rnaseq-decom_2026_05_19", "W.tsv.gz")
CLINICAL_DATA_FILE = path.join(DATASET_DIR, "12122025_Clinical export DECIDER collab.xlsx")
SV_PATH = "" #path.join(DATASET_DIR, "cohort_genedata_20261103.csv")

EXTRACTED_DATA_DIR = path.join(ROOT_DIR, "1_dataset", "extracted_raw_data", DATASET_NAME)

PREPROCESSED_DATA_DIR = path.join(ROOT_DIR, "2_preprocessing", "preprocessed_data", DATASET_NAME)

### choose GRAPH_EDGES_TYPE
GRAPH_EDGES_TYPE = 'hybrid_ppi_kegg'  # 'hybrid_ppi_kegg' (default), 'ppi', 'kegg_copathway'
PROCESSED_SAMPLES_DIR = path.join(ROOT_DIR, "3_samples_graph_creation", "processed_graphs", GRAPH_EDGES_TYPE)
PROCESSED_GRAPHS_DIR = path.join(PROCESSED_SAMPLES_DIR, DATASET_NAME)

SPLIT_DIR = path.join(ROOT_DIR, "4_train_test_split")

TRAINING_DIR = path.join(ROOT_DIR, "5_training")
TRAINED_MODELS_DIR = path.join(TRAINING_DIR, "trained_models")
### choose ENCODERS pretrained
#SAMPLE_ENCODER_FOLDER = path.join("ppi_oldgenes_MIL_platRes_PDS_12_11_2026", "GAT")
SAMPLE_ENCODER_FOLDER = path.join("hybrid_GAT_tissueType_12_11_2026", "GAT")
#SAMPLE_ENCODER_FOLDER = path.join("hybrid_MIL_platRes_PDS_fixed_18_11_2026", "GAT")
#PATIENT_ENCODER_FOLDER = path.join("ppi_oldgenes_MIL_platRes_PDS_12_11_2026", "MIL")
PATIENT_ENCODER_FOLDER = path.join("hybrid_MIL_platRes_fixed_PDS_18_11_2026", "MIL")

EXPLAINABILITY_DIR = path.join(ROOT_DIR, "6_explainability")
SAMPLE_ENCODER_EXPLAINED_DIR = path.join(TRAINED_MODELS_DIR, SAMPLE_ENCODER_FOLDER, "explained")
PATIENT_ENCODER_EXPLAINED_DIR = path.join(TRAINED_MODELS_DIR, PATIENT_ENCODER_FOLDER, "explained")

DRIVER_GENES_FILE = path.join(ROOT_DIR, "2_preprocessing", "known_driver_genes", "IntOGen-DriverGenes_OVT.tsv")
STRING_ALIASES_FILE = path.join(ROOT_DIR, 'ppi_string_files', '9606.protein.aliases.v12.0.txt')
STRING_PPI_FILE = path.join(ROOT_DIR, "ppi_string_files","9606.protein.links.v12.0.txt")

### choose if running PCA
RUN_PCA = False

#######################################################################################################################

# Nodes Features:
# None: all features available in graph
# 0: only Gene Expression
# 1: only CNV
NODE_FEATURE_INDEX = [0,1]
NODE_FEATURE_NAMES = [
    'gene_expression', # index 0
    'log2CopyNumber',  # index 1
    'nMajor',          # index 2
    'nMinor',          # index 3
    'Loh',             # index 4
    'breaksInGene',    # index 5
    'mutation_anyP',   # index 6
]


# KEEP FALSE if target = 'tissue_type'
USE_SAMPLES_METADATA = False  # samples metadata included or not in embeddings creation
METADATA_FEATURES = ['tissue']

USE_CLINICAL_DATA = False  # patients data included or not in embeddings creation
CLINICAL_FEATURES = [#'Age_at_Diagnosis',
                     'is_hrd',
                     #'BMI_at_Dg',
                     #'Treatment_strategy'
                     ]


TASK = "classification"  # 'classification', 'survival'  # TODO remove survival stuff

SAMPLES_LEVEL_TRAIN = False
PATIENTS_LEVEL_TRAIN = True

RUN_PERMUTATION_TEST = False  # resulted that the MIL model is learning

if TASK == 'classification':
    SAMPLE_TARGET_COL = 'tissue_type'  # 'tissue_type', 'tissue' (too granular), 'is_fluid' (useless, too easy)
    PATIENT_TARGET_COL = 'is_platinum_resistant'  # 'chemo_refractory_target', 'is_platinum_resistant', 'Primary_therapy_outcome'

    # PFI
    #PATIENT_TARGET_COL = 'PFI_KaplanM_allHGSC'

""""" extras """""

CONDENSATE = False
if CONDENSATE:
    PATIENTS_LEVEL_TRAIN = False

### choose eventual TARGET_GENES (None to auto select top 20)
TARGET_GENES = ['TP53', 'BRCA1', 'APOC3', 'LPL', 'ENSG00000170323']
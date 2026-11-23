[!] changes of files/folders names should be specified in the proper code sections, mainly in config_file_tree.py

################################################################
Run execute_step.sh to execute all scripts in this subdirectory.
################################################################

2_preprocessing/
|_ known_driver_genes/
|_ preprocessed_data/   (after scripts execution)
|   |_ <DATASET_NAME>/
|      |_ preprocessed_bulk_expr.tsv
|      |_ preprocessed_clinical_data.tsv
|      |_ preprocessed_cn.tsv
|      |_ preprocessed_samples_metadata.csv
|
|_ 1_clinical_preprocessing.py
|_ 1_samples_metadata_preprocessing.py
|_ 2_bulk_expr_preprocessing.py
|_ 3_cnv_preprocessing.py
|_ utils_clinical_clean_methods.py
[!] changes of files/folders names should be specified in the proper code sections, mainly in config_file_tree.py

################################################################
Run execute_step.sh to execute all scripts in this subdirectory.
################################################################

If dataset file are uploaded into project instead of referenced to external path (server):
Insert here a directory <dataset_dir_name> so that:

1_dataset/
|_ <DATASET_NAME>/
|   |_ bulk_RNA/
|   |   |_ (somewhere into one or more subdirectories, for example into set1/, set2/, ...) TPMlog2+1_gene_table.tsv
|   |   |_ rnaseq-decom/W.tsv
|   |_ clinical_file.xlsx
|   |_ ASCAT_combinedCnGenes_23Jun2025.tsv (cnv data ASCAT)
|   |_ segmentation_info_final.csv (cnv data PURPLE)
|
|_ extracted_data/   (after scripts execution)
|   |_ <DATASET_NAME>/
|       |_ clinical_data_analysis.txt
|       |_ raw_bulk_expr.tsv
|       |_ raw_clinical_data.tsv
|       |_ samples_metadata.tsv
|
|_ extract_and_analyze_clinical_data.py
|_ extract_bulk_and_metadata.py
|_ extract_cn_data.py
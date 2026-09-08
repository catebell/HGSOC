[!] changes of names must be specified in the proper code sections

Insert here a directory <dataset_dir_name> containing directories of the kind of data desired.
<bulk_RNA> should contain a file <bulk_expr.tsv> [content = matrix (genes x samples)]

Tree:

1_dataset\
|
|_ <dataset_dir_name>\
|   |
|   |_ bulk_RNA\
|       |
|       |_ bulk_expr.tsv
|       |_ W.tsv
|
|_ extracted_data\
|   |
|   |_ <dataset_dir_name>\
|       |
|       |_ bulk_expr_filtered.tsv
|       |_ samples_metadata.tsv
|
|_ 1_extract_metadata.py
|_ 2_extract_bulk.py

[!] changes of names must be specified in the proper code sections

Insert here a directory <dataset_dir_name> containing directories of the kind of data desired.
<bulk_RNA> should contain a file <bulk_expr.tsv> [content = matrix (genes x samples)]

Tree:

1_dataset\
|
|_ <dataset_dir_name>\
|   |
|   |_ <bulk_RNA>\  ->  <bulk_expr>.tsv
|
|_ extracted\
|
|_ <extract>.py  (eventual script to obtain <bulk_expr>.tsv in correct format)

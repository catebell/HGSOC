import os
import pandas as pd

import config_file_tree

sv_filepath = config_file_tree.SV_PATH
output_filepath = os.path.join(config_file_tree.PREPROCESSED_DATA_DIR, "preprocessed_sv_data.tsv")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    df_data = pd.read_csv(sv_filepath, sep='\t')
    df_data = df_data[['patient', 'gene_name', 'anyP']]

    print(f"Preprocessed structural variations file ({df_data['patient'].nunique()} patients) saved to {output_filepath}")

    df_data.to_csv(output_filepath, sep='\t', index=False)

    print("\nDONE!\n")
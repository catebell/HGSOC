from os import path, makedirs
from pandas import read_excel
from tqdm import tqdm
import re

import config_file_tree

clinical_data_filepath = config_file_tree.CLINICAL_DATA_FILE

dir_to_save = config_file_tree.EXTRACTED_DATA_DIR
makedirs(dir_to_save, exist_ok=True)

output_filepath = path.join(dir_to_save, 'raw_clinical_data.tsv')


def clean_column_name(col_name):
    """
    1) Removes any text before '::' (separator included)
    2) Replaces blank-spaces with '_'
    """
    if not isinstance(col_name, str):
        return col_name

    if '::' in col_name:
        col_name = col_name.split('::')[-1]

    col_name = col_name.strip()
    col_name = re.sub(r'\s+', '_', col_name)

    return col_name


def normalize_values_text(val):
    """
    1) Converts text to lowercase
    2) replaces '\n' and '+' with ','
    3) Removes blank-spaces around ',' and collapses multiple commas
    """
    if not isinstance(val, str):
        return val

    val = val.strip().lower()
    val = re.sub(r'[\n+]', ',', val)
    val = re.sub(r'\s*,\s*', ',', val)
    val = re.sub(r',+', ',', val).strip(' ,')

    return val


def compute_stats():
    clinical_data_df = read_excel(clinical_data_filepath)
    clinical_stats_filepath = path.join(dir_to_save, 'clinical_data_analysis.txt')

    clinical_data_df = clinical_data_df.rename(columns=clean_column_name)

    columns = clinical_data_df.select_dtypes(include=['object', 'string']).columns
    for col in columns:
        clinical_data_df[col] = clinical_data_df[col].apply(normalize_values_text)

    clinical_data_df.to_csv(output_filepath, sep='\t', index=False)
    print("Clinical file extracted saved to " + clinical_stats_filepath)

    with open(clinical_stats_filepath, 'w') as f:
        N = len(clinical_data_df)
        i=0

        for col in tqdm(clinical_data_df.columns):
            counts = len(clinical_data_df[col].value_counts())

            # all features stats listed in f:
            i=i+1
            print("\n\n" + str(i) + ") " + str(col) + ": " + str(round((N - clinical_data_df[col].isna().sum()) * 100 / N, 2)) +
                  "% of patients have this data. [Different values present: " + str(counts) + "]", file=f)
            print(clinical_data_df[col].value_counts(), file=f)

        print("Stats details saved in " + clinical_stats_filepath)


if __name__ == "__main__":
    compute_stats()
    print("\nDONE!")
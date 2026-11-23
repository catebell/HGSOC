#!/bin/bash
PYTHONPATH=.. python 1_clinical_preprocessing.py
PYTHONPATH=.. python 1_samples_metadata_preprocessing.py
PYTHONPATH=.. python 2_bulk_expr_preprocessing.py
PYTHONPATH=.. python 3_cnv_preprocessing.py
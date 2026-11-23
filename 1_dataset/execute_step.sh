#!/bin/bash
PYTHONPATH=.. python extract_and_analyze_clinical_data.py
PYTHONPATH=.. python extract_bulk_and_metadata.py
PYTHONPATH=.. python extract_cn_data.py
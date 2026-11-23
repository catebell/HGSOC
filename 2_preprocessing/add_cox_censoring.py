import argparse
import os
import numpy as np
import pandas as pd

try:
    import config_file_tree
    DEFAULT_DATA_DIR = config_file_tree.PREPROCESSED_DATA_DIR
except ImportError:
    DEFAULT_DATA_DIR = "preprocessed_data"


def add_cox_censoring_columns(input_path, output_path):
    df = pd.read_csv(input_path, sep="\t")

    prog_nd_col = "Progression_date_ND__No_prog_at_regular_follow_up_but_no_further_follow_up"
    prog_nd = df[prog_nd_col].astype(str).str.strip().str.lower()
    surv = df["Survival"].astype(str).str.strip().str.lower()
    plat_res = df["is_platinum_resistant"].astype(str).str.strip().str.lower()
    prim_out = df["Primary_therapy_outcome"].astype(str).str.strip().str.lower()
    excl_pfi = df["Excluded_from_PFI_calculations"].astype(str).str.strip().str.lower()

    # Progressione documentata: Progression_date_ND == 'no' (oppure resistenza primaria / non-risposta)
    has_progression = (prog_nd == "no") | (plat_res == "yes") | (prim_out == "unresponsive")

    # 1. PFI_event (1 = progressione/recidiva, 0 = censored libera da progressione)
    pfi_time = pd.to_numeric(df["PFI_KaplanM_allHGSC"], errors="coerce")
    valid_pfi = pfi_time.notna() & (excl_pfi != "yes")
    df["PFI_event"] = pd.Series(np.nan, index=df.index, dtype="Int64")
    df.loc[valid_pfi, "PFI_event"] = np.where(has_progression.loc[valid_pfi], 1, 0)

    # 2. PFS_event (1 = progressione o decesso per qualsiasi causa, 0 = viva senza progressione)
    pfs_time = pd.to_numeric(df["PFS_KaplanM_allHGSC"], errors="coerce")
    valid_pfs = pfs_time.notna()
    df["PFS_event"] = pd.Series(np.nan, index=df.index, dtype="Int64")
    df.loc[valid_pfs, "PFS_event"] = np.where((has_progression | (surv == "dead")).loc[valid_pfs], 1, 0)

    # 3. OS_event (1 = decesso, 0 = viva all'ultimo follow-up)
    os_time = pd.to_numeric(df["OS_KaplanM_allHGSC"], errors="coerce")
    valid_os = os_time.notna()
    df["OS_event"] = pd.Series(np.nan, index=df.index, dtype="Int64")
    df.loc[valid_os, "OS_event"] = np.where((surv == "dead").loc[valid_os], 1, 0)

    df.to_csv(output_path, sep="\t", index=False)

    # Sanity check rapido su eventi vs censored e mediana dei giorni
    print(f"File salvato in: {output_path} (righe: {len(df)})\n")
    for time_col, ev_col in [
        ("PFI_KaplanM_allHGSC", "PFI_event"),
        ("PFS_KaplanM_allHGSC", "PFS_event"),
        ("OS_KaplanM_allHGSC", "OS_event"),
    ]:
        sub = df.dropna(subset=[time_col, ev_col])
        n_ev = int((sub[ev_col] == 1).sum())
        n_cens = int((sub[ev_col] == 0).sum())
        med_ev = sub.loc[sub[ev_col] == 1, time_col].median()
        med_cens = sub.loc[sub[ev_col] == 0, time_col].median()
        print(
            f"[{ev_col}] Validi={len(sub)} | Eventi(1)={n_ev} (mediana={med_ev:.0f}g) | "
            f"Censored(0)={n_cens} (mediana={med_cens:.0f}g)"
        )

    print("\nControllo incrociato PFI_event vs is_platinum_resistant:")
    print(pd.crosstab(df["is_platinum_resistant"].fillna("NA"), df["PFI_event"].astype(str), margins=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggiunge colonne di censoring Cox a preprocessed_clinical_data.tsv")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(DEFAULT_DATA_DIR, "preprocessed_clinical_data.tsv"),
        help="Path a preprocessed_clinical_data.tsv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(DEFAULT_DATA_DIR, "preprocessed_clinical_data_cox.tsv"),
        help="Path di output per preprocessed_clinical_data_cox.tsv",
    )
    args = parser.parse_args()
    add_cox_censoring_columns(args.input, args.output)
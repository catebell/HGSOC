import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

import config_file_tree

# Mappatura automatica dei modelli supportati in oof_patient_predictions.tsv
MODEL_SCORE_SPECS = [
    ("ElasticNet", "enet_prob_resistant", "enet_pred"),
    ("PLS-DA", "pls_prob_resistant", "pls_pred"),
    ("NeuralMIL", "mil_prob_resistant", "mil_pred"),
]


def _load_oof_with_survival(model_dir, time_col="PFI_KaplanM_allHGSC", event_col=None):
    """Carica oof_patient_predictions.tsv e collega tempo all'evento e censoring dal TSV clinico."""
    oof_path = os.path.join(model_dir, "oof_patient_predictions.tsv")
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"File OOF non trovato: {oof_path}")

    df = pd.read_csv(oof_path, sep="\t")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    full_clin_path = os.path.join(config_file_tree.PREPROCESSED_DATA_DIR, "preprocessed_clinical_data.tsv")
    if os.path.exists(full_clin_path):
        df_clin = pd.read_csv(full_clin_path, sep="\t")
        df_clin["patient_id"] = df_clin["patient_id"].astype(str).str.strip()
        df_clin = df_clin.drop_duplicates("patient_id")

        # Auto-rilevamento della colonna di censoring/evento se non passata esplicitamente
        if event_col is None:
            candidate_event_cols = [
                "PFI_event", "PFI_status", "pfi_event", "pfi_status",
                "progression_event", "relapse_event", "event"
            ]
            event_col = next((c for c in candidate_event_cols if c in df_clin.columns), None)

        cols_to_merge = [c for c in [time_col, event_col] if c and c in df_clin.columns and c not in df.columns]
        if cols_to_merge:
            df = df.merge(df_clin[["patient_id"] + cols_to_merge], on="patient_id", how="left")

    if time_col not in df.columns:
        raise ValueError(f"Colonna temporale '{time_col}' assente in OOF e nei dati clinici.")

    df = df.dropna(subset=[time_col]).copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df[df[time_col] >= 0].copy()

    # Se non esiste una colonna di censoring esplicita, assume eventi osservati (1)
    if event_col and event_col in df.columns:
        df["_event"] = pd.to_numeric(df[event_col], errors="coerce").fillna(1).astype(int)
    else:
        print("[INFO] Nessuna colonna di censoring PFI rilevata: imposto event=1 per tutte le pazienti con PFI noto.")
        df["_event"] = 1

    return df


def _bootstrap_cindex(times, risk_scores, events, n_boot=1000, seed=42):
    """Calcola C-index (con -risk_scores perche' rischio alto = PFI breve) e CI 95% bootstrap."""
    c_obs = float(concordance_index(times, -risk_scores, events))
    rng = np.random.default_rng(seed)
    n = len(times)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(events[idx])) < 1 or np.std(risk_scores[idx]) == 0:
            continue
        try:
            boot_vals.append(concordance_index(times[idx], -risk_scores[idx], events[idx]))
        except ZeroDivisionError:
            continue
    ci_low, ci_high = (np.percentile(boot_vals, [2.5, 97.5]) if boot_vals else (np.nan, np.nan))
    return c_obs, float(ci_low), float(ci_high)


def run_posthoc_cox_oof_validation(model_dir, time_col="PFI_KaplanM_allHGSC", event_col=None):
    """Esegue Cox PH univariata, C-index e Kaplan-Meier sugli score OOF."""
    df = _load_oof_with_survival(model_dir, time_col=time_col, event_col=event_col)
    out_dir = os.path.join(model_dir, "survival_cox_oof")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    active_models = [(name, p_col, pred_col) for name, p_col, pred_col in MODEL_SCORE_SPECS if p_col in df.columns]

    if not active_models:
        raise ValueError(f"Nessuna colonna di probabilita' trovata in {model_dir}/oof_patient_predictions.tsv")

    fig, axes = plt.subplots(len(active_models), 2, figsize=(13, 5 * len(active_models)), squeeze=False)

    for row_idx, (model_name, prob_col, pred_col) in enumerate(active_models):
        sub = df.dropna(subset=[prob_col]).copy()
        probs = np.clip(sub[prob_col].values.astype(float), 1e-5, 1.0 - 1e-5)
        times = sub[time_col].values.astype(float)
        events = sub["_event"].values.astype(int)

        # 1. Trasformazioni dello score:
        #    - prob_z: probabilita' standardizzata (HR per +1 SD)
        #    - logit: log(p / (1-p)), lineare nello spazio dei log-hazard
        #    - risk_high: stratificazione binaria (pred == 1 oppure mediana se pred assente)
        sub["prob_z"] = (probs - probs.mean()) / max(probs.std(), 1e-8)
        sub["logit_score"] = np.log(probs / (1.0 - probs))
        if pred_col in sub.columns:
            sub["risk_high"] = sub[pred_col].astype(int)
        else:
            sub["risk_high"] = (probs >= np.median(probs)).astype(int)

        # 2. C-index globale OOF + Bootstrap 95% CI
        c_idx, c_low, c_high = _bootstrap_cindex(times, probs, events)

        # 3. Fit modelli di Cox univariati
        for feat_col, feat_label in [
            ("prob_z", "OOF Probability (per +1 SD)"),
            ("logit_score", "OOF Logit Score (log-odds)"),
            ("risk_high", "Predicted High-Risk Group (1 vs 0)"),
        ]:
            cph = CoxPHFitter()
            cph.fit(sub[[time_col, "_event", feat_col]], duration_col=time_col, event_col="_event")
            s = cph.summary.loc[feat_col]

            summary_rows.append({
                "model": model_name,
                "n_patients": len(sub),
                "n_events": int(events.sum()),
                "c_index": round(c_idx, 4),
                "c_index_ci95_low": round(c_low, 4),
                "c_index_ci95_high": round(c_high, 4),
                "predictor": feat_label,
                "coef_beta": round(float(s["coef"]), 4),
                "HR": round(float(s["exp(coef)"]), 4),
                "HR_ci95_low": round(float(s["exp(coef) lower 95%"]), 4),
                "HR_ci95_high": round(float(s["exp(coef) upper 95%"]), 4),
                "cox_p_value": float(s["p"]),
            })

        # 4. Log-rank test tra gruppi di rischio
        grp_hi = sub[sub["risk_high"] == 1]
        grp_lo = sub[sub["risk_high"] == 0]
        lr_res = logrank_test(
            grp_hi[time_col], grp_lo[time_col],
            event_observed_A=grp_hi["_event"], event_observed_B=grp_lo["_event"]
        )
        hr_bin = summary_rows[-1]["HR"]
        hr_bin_ci = f"{summary_rows[-1]['HR_ci95_low']:.2f}-{summary_rows[-1]['HR_ci95_high']:.2f}"

        # --- Pannello A: Curve di Kaplan-Meier ---
        ax_km = axes[row_idx, 0]
        kmf = KaplanMeierFitter()
        for grp_df, label, color in [
            (grp_lo, f"Predicted Sensitive (n={len(grp_lo)})", "#0d9488"),
            (grp_hi, f"Predicted Resistant (n={len(grp_hi)})", "#be185d"),
        ]:
            if not grp_df.empty:
                kmf.fit(grp_df[time_col], event_observed=grp_df["_event"], label=label)
                kmf.plot_survival_function(ax=ax_km, ci_show=True, color=color, linewidth=2.2)

        is_pfi = "pfi" in time_col.lower()
        endpoint_short = "PFI" if is_pfi else ("OS" if "os_" in time_col.lower() else "PFS")
        time_label = {
            "PFI": "Platinum-Free Interval (Days)",
            "PFS": "Progression-Free Survival (Days)",
            "OS": "Overall Survival (Days)",
        }[endpoint_short]
        prob_label = "Survival Probability" if endpoint_short == "OS" else "Progression-Free Probability"

        # Nel pannello KM (ax_km):
        if is_pfi:
            ax_km.axvline(180, color="gray", linestyle=":", linewidth=1.1, label="180d Clinical Cutoff")
        ax_km.set_title(
            f"{model_name} OOF — Kaplan-Meier {endpoint_short}\n"
            f"Log-rank p = {lr_res.p_value:.4g} | HR = {hr_bin:.2f} (95% CI: {hr_bin_ci})"
        )
        ax_km.set_xlabel(time_label)
        ax_km.set_ylabel(prob_label)
        ax_km.grid(True, alpha=0.25)
        ax_km.legend(loc="best", fontsize=8.5)

        # --- Pannello B: Score continuo OOF vs PFI ---
        ax_sc = axes[row_idx, 1]
        colors_pt = np.where(sub["target_label"] == 1, "#be185d", "#0d9488")
        ax_sc.scatter(probs, times, c=colors_pt, alpha=0.85, edgecolors="k", s=55)
        if is_pfi:
            ax_sc.axhline(180, color="gray", linestyle=":", linewidth=1.1, label="180d Resistance Cutoff")
        ax_sc.set_title(
            f"{model_name} OOF — Continuous Risk vs. {endpoint_short}\n"
            f"C-index = {c_idx:.3f} (95% CI: {c_low:.3f}-{c_high:.3f})"
        )
        ax_sc.set_ylabel(time_label)
        ax_sc.grid(True, alpha=0.25)

    df_summary = pd.DataFrame(summary_rows)
    tsv_out = os.path.join(out_dir, "cox_oof_prognostic_summary.tsv")
    df_summary.to_csv(tsv_out, sep="\t", index=False)

    plt.tight_layout()
    png_out = os.path.join(out_dir, "cox_oof_km_and_risk_score.png")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n=== POST-HOC COX OOF VALIDATION ({os.path.basename(model_dir)}) ===")
    print(df_summary.to_string(index=False))
    print(f"\nSalvato report TSV: {tsv_out}")
    print(f"Salvato grafico KM/Cox: {png_out}\n")
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc Cox & C-index validation on OOF predictions")
    parser.add_argument("--model_dir", type=str, required=True, help="Path alla cartella LinearMIL/ o MIL/")
    parser.add_argument("--time_col", type=str, default="PFI_KaplanM_allHGSC")
    parser.add_argument("--event_col", type=str, default=None, help="Colonna evento/censoring (opzionale)")
    args = parser.parse_args()
    run_posthoc_cox_oof_validation(args.model_dir, time_col=args.time_col, event_col=args.event_col)
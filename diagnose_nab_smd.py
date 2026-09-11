import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generalization_v2.evaluate_real_benchmarks import (
    BaselineConvAE,
    TinyConvAE,
    USAD,
    AnomalyTransformer,
    PatchTSTBackbone,
    _batched_forward_scores,
    smooth_errors,
    dynamic_threshold_channel,
    point_adjust,
    compute_affiliation_metrics
)
from external_benchmark.run_external_benchmark import load_nab, load_checkpoint_strict

def diagnose_nab():
    print("=" * 80)
    print("      DIAGNOSTIC: USAD TEACHER vs. DISTILLED STUDENT ON NAB (58 SERIES)")
    print("=" * 80)

    nab_root = os.path.join(PROJECT_ROOT, "external_benchmark", "data_external", "nab")
    entities = load_nab(nab_root)
    print(f"Total NAB entities loaded: {len(entities)}")

    v2_ckpt_dir = os.path.join(PROJECT_ROOT, "generalization_v2", "checkpoints")
    
    usad_model = load_checkpoint_strict(USAD(window_size=100, n_features=1), os.path.join(v2_ckpt_dir, "usad_teacher_v2.pth"))
    student_model = load_checkpoint_strict(TinyConvAE(n_features=1), os.path.join(v2_ckpt_dir, "student_v2.pth"))

    usad_raw_f1s = []
    student_raw_f1s = []

    series_comparisons = []

    for ent in entities:
        name = ent["entity_id"]
        Xw, yw = ent["X_windows"], ent["y_windows"]
        if yw.sum() == 0:
            continue

        # USAD
        sc_u = _batched_forward_scores(usad_model, "USAD", Xw, batch_size=512)
        sm_u = smooth_errors(sc_u)
        th_u = dynamic_threshold_channel(sm_u)
        pred_u = (sm_u > th_u).astype(int)
        f1_u = f1_score(yw, pred_u, zero_division=0)
        usad_raw_f1s.append(f1_u)

        # Student
        sc_s = _batched_forward_scores(student_model, "TinyConvAE", Xw, batch_size=512)
        sm_s = smooth_errors(sc_s)
        th_s = dynamic_threshold_channel(sm_s)
        pred_s = (sm_s > th_s).astype(int)
        f1_s = f1_score(yw, pred_s, zero_division=0)
        student_raw_f1s.append(f1_s)

        series_comparisons.append({
            "series": os.path.basename(name),
            "anom_win": int(yw.sum()),
            "total_win": len(yw),
            "usad_thresh": round(float(th_u), 4),
            "usad_flagged": int(pred_u.sum()),
            "usad_f1": round(float(f1_u), 6),
            "student_thresh": round(float(th_s), 4),
            "student_flagged": int(pred_s.sum()),
            "student_f1": round(float(f1_s), 6),
            "diff": round(float(f1_u - f1_s), 6)
        })

    df_comp = pd.DataFrame(series_comparisons)
    print(f"\nAnalyzed {len(df_comp)} anomaly-bearing NAB series.")
    print(f"USAD Mean Raw-F1 (unrounded):    {np.mean(usad_raw_f1s):.8f}")
    print(f"Student Mean Raw-F1 (unrounded): {np.mean(student_raw_f1s):.8f}")
    print(f"Series where F1 differs:         {(df_comp['diff'] != 0).sum()} / {len(df_comp)}")
    
    print("\nSample Series Comparison (where models differ):")
    diff_series = df_comp[df_comp["diff"] != 0]
    print(diff_series.head(15).to_string(index=False))

    print("\nSample Series Comparison (where models tie at zero or non-zero):")
    same_series = df_comp[df_comp["diff"] == 0]
    print(same_series.head(10).to_string(index=False))

if __name__ == "__main__":
    diagnose_nab()

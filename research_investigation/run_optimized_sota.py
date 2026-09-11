"""
research_investigation/run_optimized_sota.py
--------------------------------------------
Evaluates the true SOTA configurations based on the scientific ablation findings:
  1. Pure Reconstruction MSE (eliminating the first-difference jitter penalty)
  2. Optimal temporal granularity (W=64, Stride=5 vs W=100, Stride=10)
  3. Frozen Validation Threshold Calibration (tau* = argmax_tau F1_val(tau))
  4. MultiScaleStudent-911p (Parallel k=3, 7, 15) vs Baseline TinyConvAE (421p)
  5. Multi-Domain evaluation across NASA SMAP/MSL, OPS-SAT-AD, SMD, and NAB
"""

import os
import sys
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "research_investigation")
RESULTS_TABLES = os.path.join(OUT_DIR, "results", "tables")
RESULTS_FIGS = os.path.join(OUT_DIR, "results", "figures")
os.makedirs(RESULTS_TABLES, exist_ok=True)
os.makedirs(RESULTS_FIGS, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from research_investigation.run_controlled_ablations import (
    StandardTinyConvAE,
    MultiScaleStudent911p,
    prepare_dataset,
    train_model,
    score_windows,
    smooth_errors,
    calibrate_threshold,
    point_adjust,
    compute_affiliation_metrics
)

def evaluate_optimized_pipeline():
    print("=" * 80)
    print("      EVALUATING OPTIMIZED PIPELINE (NO FIRST-DIFF PENALTY)")
    print("=" * 80)
    
    # 1. Load Data with W=64 and W=100
    chans_w64 = prepare_dataset(window_size=64, stride=5)
    chans_w100 = prepare_dataset(window_size=100, stride=10)
    
    all_tr_w64 = np.concatenate([c["train_win"] for c in chans_w64[:15]], axis=0)
    all_tr_w100 = np.concatenate([c["train_win"] for c in chans_w100[:15]], axis=0)
    
    # Evaluate 4 key models with Pure Recon MSE + Calibrated Threshold
    models_to_test = [
        {"name": "TinyConvAE (421p, W=100, mu+3sigma)", "model": StandardTinyConvAE(), "w": 100, "chans": chans_w100, "X_tr": all_tr_w100, "calib": False, "params": 421},
        {"name": "TinyConvAE (421p, W=64, tau* Calibrated)", "model": StandardTinyConvAE(), "w": 64, "chans": chans_w64, "X_tr": all_tr_w64, "calib": True, "params": 421},
        {"name": "MultiScaleStudent-911p (911p, W=100, tau* Calibrated)", "model": MultiScaleStudent911p(), "w": 100, "chans": chans_w100, "X_tr": all_tr_w100, "calib": True, "params": 911},
        {"name": "MultiScaleStudent-911p (911p, W=64, tau* Calibrated)", "model": MultiScaleStudent911p(), "w": 64, "chans": chans_w64, "X_tr": all_tr_w64, "calib": True, "params": 911},
    ]
    
    records = []
    
    for m_info in models_to_test:
        print(f"\nTraining {m_info['name']}...")
        model = train_model(m_info["model"], m_info["X_tr"], epochs=25)
        
        raw_f1s, aff_f1s, pa_f1s, precs, recs = [], [], [], [], []
        
        for ch in m_info["chans"]:
            if ch["test_y"].sum() == 0:
                continue
            
            # Val threshold
            val_sc = score_windows(model, ch["val_win"], use_composite=False)
            val_sm = smooth_errors(val_sc)
            
            if m_info["calib"]:
                tau = calibrate_threshold(val_sm, ch["val_y"])
            else:
                tau = val_sm.mean() + 3.0 * val_sm.std()
                
            # Test evaluation
            test_sc = score_windows(model, ch["test_win"], use_composite=False)
            test_sm = smooth_errors(test_sc)
            preds = (test_sm > tau).astype(int)
            pa_preds = point_adjust(ch["test_y"], preds)
            
            raw_f1s.append(f1_score(ch["test_y"], preds, zero_division=0))
            aff_res = compute_affiliation_metrics(ch["test_y"], preds)
            aff_f1s.append(aff_res["aff_f1"])
            pa_f1s.append(f1_score(ch["test_y"], pa_preds, zero_division=0))
            precs.append(precision_score(ch["test_y"], preds, zero_division=0))
            recs.append(recall_score(ch["test_y"], preds, zero_division=0))
            
        m_raw = float(np.mean(raw_f1s)) if raw_f1s else 0.0
        m_aff = float(np.mean(aff_f1s)) if aff_f1s else 0.0
        m_pa = float(np.mean(pa_f1s)) if pa_f1s else 0.0
        m_prec = float(np.mean(precs)) if precs else 0.0
        m_rec = float(np.mean(recs)) if recs else 0.0
        
        records.append({
            "Model Pipeline": m_info["name"],
            "Parameters": m_info["params"],
            "Footprint": f"{(m_info['params'] * 4) / 1024:.2f} KB",
            "Precision": round(m_prec, 4),
            "Recall": round(m_rec, 4),
            "Raw-F1": round(m_raw, 4),
            "Affiliation-F1": round(m_aff, 4),
            "Point-Adjusted F1": round(m_pa, 4)
        })
        print(f"  --> Raw-F1: {m_raw:.4f} | Aff-F1: {m_aff:.4f} | PA-F1: {m_pa:.4f} | Prec: {m_prec:.4f} | Rec: {m_rec:.4f}")
        
    df_res = pd.DataFrame(records)
    csv_out = os.path.join(RESULTS_TABLES, "optimized_pipeline_comparison.csv")
    tex_out = os.path.join(RESULTS_TABLES, "optimized_pipeline_comparison.tex")
    df_res.to_csv(csv_out, index=False)
    df_res.to_latex(tex_out, index=False, caption="Optimized Pipeline Comparison: Impact of Multi-Scale Kernels, Window Length, and Validation Calibration.")
    
    # Plot Figure: Comparison of Pipelines
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(df_res))
    w = 0.25
    plt.bar(x - w, df_res["Raw-F1"], width=w, label="Raw-F1 (Strict)", color="#2563eb")
    plt.bar(x, df_res["Affiliation-F1"], width=w, label="Affiliation-F1", color="#059669")
    plt.bar(x + w, df_res["Point-Adjusted F1"], width=w, label="Point-Adjusted F1", color="#d97706")
    plt.xticks(x, ["Baseline\n(421p, W100, Heur)", "Calibrated\n(421p, W64, Calib)", "MultiScale-911p\n(W100, Calib)", "MultiScale-911p\n(W64, Calib)"], fontsize=9)
    plt.ylabel("F1 Score", fontsize=11)
    plt.title("Optimized Pipeline Benchmark (NASA SMAP/MSL)", fontsize=12, fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_FIGS, "optimized_pipeline_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print("\n" + "=" * 80)
    print("      OPTIMIZED PIPELINE RESULTS")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print(f"\n[Saved CSV]    {csv_out}")
    print(f"[Saved LaTeX]  {tex_out}")
    print(f"[Saved Figure] {fig_path}")
    return df_res

if __name__ == "__main__":
    evaluate_optimized_pipeline()

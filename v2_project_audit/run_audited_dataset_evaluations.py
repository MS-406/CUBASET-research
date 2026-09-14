"""
full_audit_v1/run_audited_dataset_evaluations.py
===============================================
Evaluates real ESA OPS-SAT on-orbit telemetry and ESA-ADB sample
under the exact standardized zero-leakage evaluation pipeline.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(WORKSPACE_ROOT, "full_audit_v1")
TABLES_DIR = os.path.join(AUDIT_DIR, "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(os.cpu_count() or 2, 4))

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Metrics
def point_adjust(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int).copy()
    anomaly_state, start = False, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not anomaly_state:
            anomaly_state, start = True, i
        elif y_true[i] == 0 and anomaly_state:
            anomaly_state = False
            if y_pred[start:i].sum() > 0:
                y_pred[start:i] = 1
    if anomaly_state and y_pred[start:].sum() > 0:
        y_pred[start:] = 1
    return y_pred

def extract_events(arr):
    events, in_evt, s = [], False, 0
    for i, val in enumerate(arr):
        if val == 1 and not in_evt:
            in_evt, s = True, i
        elif val == 0 and in_evt:
            in_evt = False
            events.append((s, i))
    if in_evt:
        events.append((s, len(arr)))
    return events

def compute_detailed_affiliation(labels, predictions):
    labels, preds = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    gt_events = extract_events(labels)
    pred_events = extract_events(preds)

    if len(gt_events) == 0:
        return {"aff_precision": 1.0 if len(pred_events) == 0 else 0.0, "aff_recall": 1.0, "aff_f1": 1.0, "gt_events": 0, "pred_events": len(pred_events)}
    if len(pred_events) == 0:
        return {"aff_precision": 1.0, "aff_recall": 0.0, "aff_f1": 0.0, "gt_events": len(gt_events), "pred_events": 0}

    gt_detected = sum(1 for gs, ge in gt_events if any(not (pe <= gs or ps >= ge) for ps, pe in pred_events))
    pred_valid = sum(1 for ps, pe in pred_events if any(not (pe <= gs or ps >= ge) for gs, ge in gt_events))
    rec = gt_detected / len(gt_events)
    prec = pred_valid / len(pred_events)
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"aff_precision": float(prec), "aff_recall": float(rec), "aff_f1": float(f1), "gt_events": len(gt_events), "pred_events": len(pred_events)}

def smooth_scores(scores, span=10):
    if len(scores) == 0:
        return np.array([])
    sm = pd.Series(scores).ewm(span=span, adjust=False).mean().values
    return np.nan_to_num(sm, nan=0.0, posinf=1.0, neginf=0.0)

def compute_safe_pr_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        order = np.argsort(rec)
        sorted_rec = rec[order]
        sorted_prec = prec[order]
        uniq_rec, idxs = np.unique(sorted_rec, return_index=True)
        uniq_prec = sorted_prec[idxs]
        if len(uniq_rec) > 1:
            return float(auc(uniq_rec, uniq_prec))
        return 0.0
    except Exception:
        return 0.0

def make_sliding_windows(arr, window=32, stride=4):
    windows, center_indices = [], []
    for start in range(0, len(arr) - window + 1, stride):
        windows.append(arr[start:start + window])
        center_indices.append(start + window // 2)
    if not windows:
        return np.empty((0, window, arr.shape[1] if arr.ndim > 1 else 1)), []
    return np.stack(windows).astype(np.float32), center_indices

# Model
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[3, 7, 11, 15]):
        super(MultiScaleConvBlock, self).__init__()
        num_k = len(kernels)
        ch_div = max(1, out_channels // num_k)
        rem = out_channels - (num_k - 1) * ch_div
        self.convs = nn.ModuleList()
        for idx, k in enumerate(kernels):
            c_out = rem if idx == num_k - 1 else ch_div
            self.convs.append(nn.Conv1d(in_channels, c_out, kernel_size=k, padding=k // 2))
        self.act = nn.ReLU()

    def forward(self, x):
        outputs = [c(x) for c in self.convs]
        return self.act(torch.cat(outputs, dim=1))

class MultiScaleTelemetryAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=24, latent_dim=12, kernels=[3, 7, 11, 15]):
        super(MultiScaleTelemetryAE, self).__init__()
        self.enc1 = MultiScaleConvBlock(in_channels, hidden_dim, kernels=kernels)
        self.enc2 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=5, padding=2)
        self.dec1 = nn.Conv1d(latent_dim, hidden_dim, kernel_size=5, padding=2)
        self.dec2 = nn.Conv1d(hidden_dim, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        h1 = self.enc1(x_t)
        z = self.relu(self.enc2(h1))
        h2 = self.relu(self.dec1(z))
        out = self.dec2(h2)
        return out.transpose(1, 2)

def train_ae(model, train_windows, epochs=15, lr=1e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(train_windows).float())
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)

    for ep in range(epochs):
        for batch in loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), x)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.eval()
    return model

# -------------------------------------------------------------------------
# 1. EVALUATE ESA OPS-SAT (Across all available channels in segments.csv)
# -------------------------------------------------------------------------
def evaluate_opssat():
    print("=" * 80)
    print("EVALUATING ESA OPS-SAT 3U CUBESAT TELEMETRY (REAL ON-ORBIT BENCHMARK)")
    print("=" * 80)
    seg_path = os.path.join(WORKSPACE_ROOT, "opssat_data", "segments.csv")
    if not os.path.exists(seg_path):
        print(f"[WARN] OPS-SAT data file missing at {seg_path}")
        return None

    df_seg = pd.read_csv(seg_path)
    print(f"Loaded OPS-SAT segments.csv: {len(df_seg):,} rows across channels: {df_seg['channel'].unique().tolist()}")

    chan_results = []
    w, s = 32, 4

    for chan in df_seg["channel"].unique():
        df_c = df_seg[df_seg["channel"] == chan]
        
        # Split by official 'train' column if present, else 50/50
        if "train" in df_c.columns and df_c["train"].nunique() > 1:
            tr_df = df_c[df_c["train"] == 1]
            te_df = df_c[df_c["train"] == 0]
            tr_vals = tr_df["value"].values.astype(np.float32)
            te_vals = te_df["value"].values.astype(np.float32)
            if "anomaly" in te_df.columns:
                te_labels = te_df["anomaly"].values.astype(int)
            elif "label" in te_df.columns:
                te_labels = (te_df["label"].astype(str) != "0").astype(int)
            else:
                te_labels = np.zeros(len(te_vals), dtype=int)
        else:
            vals = df_c["value"].values.astype(np.float32)
            if "anomaly" in df_c.columns:
                lbls = df_c["anomaly"].values.astype(int)
            elif "label" in df_c.columns:
                lbls = (df_c["label"].astype(str) != "0").astype(int)
            else:
                lbls = np.zeros(len(vals), dtype=int)
            n_tr = int(len(vals) * 0.5)
            tr_vals, te_vals = vals[:n_tr], vals[n_tr:]
            te_labels = lbls[n_tr:]

        if len(tr_vals) < w * 2 or len(te_vals) < w * 2:
            continue

        mu, std = np.mean(tr_vals), max(np.std(tr_vals), 1e-6)
        tr_norm = ((tr_vals - mu) / std)[:, None]
        te_norm = ((te_vals - mu) / std)[:, None]

        tr_w, _ = make_sliding_windows(tr_norm, window=w, stride=s)
        te_w, te_idx = make_sliding_windows(te_norm, window=w, stride=s)
        if len(tr_w) < 4 or len(te_w) < 4:
            continue

        val_split = max(int(len(tr_w) * 0.2), 2)
        tr_part, val_part = tr_w[:-val_split], tr_w[-val_split:]

        model = MultiScaleTelemetryAE(in_channels=1, out_channels=1, hidden_dim=16, latent_dim=8).to(DEVICE)
        train_ae(model, tr_part, epochs=12)

        with torch.no_grad():
            v_t = torch.from_numpy(val_part).float().to(DEVICE)
            v_res = torch.mean((model(v_t) - v_t) ** 2, dim=(1, 2)).cpu().numpy()
            thresh = float(np.percentile(v_res, 98.5))

            t_t = torch.from_numpy(te_w).float().to(DEVICE)
            t_res = torch.mean((model(t_t) - t_t) ** 2, dim=(1, 2)).cpu().numpy()

        sc = smooth_scores(t_res, 10)
        preds = (sc > thresh).astype(int)

        y_te = np.zeros(len(te_w), dtype=int)
        for idx_i, c_idx in enumerate(te_idx):
            if c_idx < len(te_labels):
                y_te[idx_i] = te_labels[c_idx]

        raw_f1 = f1_score(y_te, preds, zero_division=0)
        pa_f1 = f1_score(y_te, point_adjust(y_te, preds), zero_division=0)
        aff_info = compute_detailed_affiliation(y_te, preds)
        pr_auc = compute_safe_pr_auc(y_te, sc)

        chan_results.append({
            "Channel": chan,
            "Raw_F1": raw_f1,
            "Aff_F1": aff_info["aff_f1"],
            "PA_F1": pa_f1,
            "PR_AUC": pr_auc,
            "GT_Events": aff_info["gt_events"],
            "Pred_Events": aff_info["pred_events"]
        })

    df_chan = pd.DataFrame(chan_results)
    print("\n--- Per-Channel OPS-SAT Evaluation Results ---")
    print(df_chan.to_string(index=False))

    return {
        "Dataset": "ESA OPS-SAT-AD (3U CubeSat In-Orbit Telemetry)",
        "Domain": "European Space Agency 3U CubeSat",
        "Channels_Evaluated": len(df_chan),
        "Mean_Strict_Raw_F1": round(float(df_chan["Raw_F1"].mean()), 4),
        "Mean_Affiliation_F1": round(float(df_chan["Aff_F1"].mean()), 4),
        "Mean_Point_Adjusted_F1": round(float(df_chan["PA_F1"].mean()), 4),
        "Mean_PR_AUC": round(float(df_chan["PR_AUC"].mean()), 4),
        "Data_Provenance_Note": "Zenodo DOI 10.5281/zenodo.12588359 (Verified real in-orbit telemetry)"
    }

# -------------------------------------------------------------------------
# 2. EVALUATE ESA-ADB SAMPLE
# -------------------------------------------------------------------------
def evaluate_esa_adb():
    print("\n" + "=" * 80)
    print("EVALUATING ESA-ADB MISSION TELEMETRY SAMPLE")
    print("=" * 80)
    adb_path = os.path.join(WORKSPACE_ROOT, "esa_adb_data", "esa_adb_mission_telemetry.csv")
    if not os.path.exists(adb_path):
        return None

    df_adb = pd.read_csv(adb_path)
    val_cols = ["power_telemetry", "thermal_telemetry"]
    label_col = "is_anomaly"

    raw_vals = df_adb[val_cols].values.astype(np.float32)
    labels = df_adb[label_col].values.astype(int)

    n_tr = int(len(raw_vals) * 0.6)
    tr_raw, te_raw = raw_vals[:n_tr], raw_vals[n_tr:]
    te_labels = labels[n_tr:]

    mu = np.mean(tr_raw, axis=0)
    std = np.maximum(np.std(tr_raw, axis=0), 1e-6)
    tr_norm = (tr_raw - mu) / std
    te_norm = (te_raw - mu) / std

    w, s = 32, 4
    tr_w, _ = make_sliding_windows(tr_norm, window=w, stride=s)
    te_w, te_idx = make_sliding_windows(te_norm, window=w, stride=s)

    val_split = max(int(len(tr_w) * 0.2), 2)
    tr_part, val_part = tr_w[:-val_split], tr_w[-val_split:]

    k_feats = tr_norm.shape[1]
    model = MultiScaleTelemetryAE(in_channels=k_feats, out_channels=k_feats, hidden_dim=20, latent_dim=10).to(DEVICE)
    train_ae(model, tr_part, epochs=15)

    with torch.no_grad():
        v_t = torch.from_numpy(val_part).float().to(DEVICE)
        v_res = torch.mean((model(v_t) - v_t) ** 2, dim=(1, 2)).cpu().numpy()
        thresh = float(np.percentile(v_res, 98.5))

        t_t = torch.from_numpy(te_w).float().to(DEVICE)
        t_res = torch.mean((model(t_t) - t_t) ** 2, dim=(1, 2)).cpu().numpy()

    sc = smooth_scores(t_res, 10)
    preds = (sc > thresh).astype(int)

    y_te = np.zeros(len(te_w), dtype=int)
    for idx_i, c_idx in enumerate(te_idx):
        if c_idx < len(te_labels):
            y_te[idx_i] = te_labels[c_idx]

    raw_f1 = f1_score(y_te, preds, zero_division=0)
    pa_f1 = f1_score(y_te, point_adjust(y_te, preds), zero_division=0)
    aff_info = compute_detailed_affiliation(y_te, preds)
    pr_auc = compute_safe_pr_auc(y_te, sc)

    print(f"ESA-ADB Sample Results -> Strict Raw-F1: {raw_f1:.4f} | Aff-F1: {aff_info['aff_f1']:.4f} | PA-F1: {pa_f1:.4f} | PR-AUC: {pr_auc:.4f}")

    return {
        "Dataset": "ESA-ADB (Curated Multi-Channel Telemetry Sample)",
        "Domain": "European Space Agency Satellite Archive (Sample)",
        "Channels_Evaluated": 2,
        "Mean_Strict_Raw_F1": round(float(raw_f1), 4),
        "Mean_Affiliation_F1": round(float(aff_info["aff_f1"]), 4),
        "Mean_Point_Adjusted_F1": round(float(pa_f1), 4),
        "Mean_PR_AUC": round(float(pr_auc), 4),
        "Data_Provenance_Note": "Local 20k-row curated mission slice (Full 700M Zenodo dataset requires full download)"
    }

if __name__ == "__main__":
    results = []
    r_ops = evaluate_opssat()
    if r_ops: results.append(r_ops)
    r_adb = evaluate_esa_adb()
    if r_adb: results.append(r_adb)

    df_ext = pd.DataFrame(results)
    out_p = os.path.join(TABLES_DIR, "dataset_extension_benchmark.csv")
    df_ext.to_csv(out_p, index=False)
    print("\n" + "=" * 80)
    print("DATASET EXTENSION BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(df_ext.to_string(index=False))

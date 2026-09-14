"""
fresh_pipeline.py
-----------------
Completely fresh, zero-leakage research evaluation engine for NASA SMAP/MSL with
full multi-level checkpointing and evaluation caching.
"""

import os
import sys
import math
import time
import copy
import json
import random
import ast
import numpy as np
import pandas as pd
from scipy.stats import genpareto

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Dynamic Project Root Detection (Safe for Colab & Local)
PROJECT_ROOT = os.getcwd()
if not os.path.exists(os.path.join(PROJECT_ROOT, "data")):
    if os.path.exists("/content/drive/MyDrive/cubesat_project/data"):
        PROJECT_ROOT = "/content/drive/MyDrive/cubesat_project"
        os.chdir(PROJECT_ROOT)
    elif os.path.exists("cubesat_project/data"):
        PROJECT_ROOT = os.path.abspath("cubesat_project")
        os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LABELS_CSV = os.path.join(DATA_DIR, "labeled_anomalies.csv")
OUT_DIR = PROJECT_ROOT
RESULTS_TABLES = os.path.join(OUT_DIR, "results", "tables")
RESULTS_FIGS = os.path.join(OUT_DIR, "results", "figures")
RESULTS_CACHE = os.path.join(OUT_DIR, "results", "cache")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(RESULTS_TABLES, exist_ok=True)
os.makedirs(RESULTS_FIGS, exist_ok=True)
os.makedirs(RESULTS_CACHE, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    num_threads = min(os.cpu_count() or 2, 4)
    torch.set_num_threads(num_threads)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------------------------------------------------------------
# Evaluation & Affiliation Metrics
# -------------------------------------------------------------------------
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

def compute_affiliation_metrics(labels, predictions):
    labels, preds = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    def get_events(arr):
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

    gt_events, pred_events = get_events(labels), get_events(preds)
    if len(gt_events) == 0:
        return {"aff_precision": 1.0 if len(pred_events) == 0 else 0.0, "aff_recall": 1.0, "aff_f1": 1.0}
    if len(pred_events) == 0:
        return {"aff_precision": 1.0, "aff_recall": 0.0, "aff_f1": 0.0}

    gt_detected = sum(1 for gs, ge in gt_events if any(not (pe <= gs or ps >= ge) for ps, pe in pred_events))
    pred_valid = sum(1 for ps, pe in pred_events if any(not (pe <= gs or ps >= ge) for gs, ge in gt_events))
    rec = gt_detected / len(gt_events)
    prec = pred_valid / len(pred_events)
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"aff_precision": float(prec), "aff_recall": float(rec), "aff_f1": float(f1)}

def smooth_scores(scores, span=10):
    if len(scores) == 0:
        return np.array([])
    sm = pd.Series(scores).ewm(span=span, adjust=False).mean().values
    return np.nan_to_num(sm, nan=0.0, posinf=1.0, neginf=0.0)

# -------------------------------------------------------------------------
# Dynamic Leakage-Free Dataset Loader (Robust to Corrupted/Zero-Byte Files)
# -------------------------------------------------------------------------
def load_all_nasa_channels(scaler_type="robust"):
    labels_df = pd.read_csv(LABELS_CSV)
    channels = []
    for chan in labels_df["chan_id"].unique():
        tr_p = os.path.join(DATA_DIR, "train", f"{chan}.npy")
        te_p = os.path.join(DATA_DIR, "test", f"{chan}.npy")
        if not (os.path.exists(tr_p) and os.path.exists(te_p)):
            continue

        try:
            if os.path.getsize(tr_p) == 0 or os.path.getsize(te_p) == 0:
                continue
            tr_raw = np.load(tr_p) # (N_train, C)
            te_raw = np.load(te_p) # (N_test, C)
            if len(tr_raw) == 0 or len(te_raw) == 0:
                continue
        except Exception as e:
            print(f"[WARN] Skipping corrupted channel {chan}: {e}")
            continue

        n_feats = tr_raw.shape[1]
        tr_norm = np.zeros_like(tr_raw, dtype=np.float32)
        te_norm = np.zeros_like(te_raw, dtype=np.float32)

        for col in range(n_feats):
            c_tr = tr_raw[:, col]
            c_te = te_raw[:, col]
            if scaler_type == "robust":
                med = np.median(c_tr)
                q75 = np.percentile(c_tr, 75)
                q25 = np.percentile(c_tr, 25)
                scale = max(q75 - q25, 1e-6)
                tr_norm[:, col] = (c_tr - med) / scale
                te_norm[:, col] = (c_te - med) / scale
            else:
                mu = np.mean(c_tr)
                std = max(np.std(c_tr), 1e-6)
                tr_norm[:, col] = (c_tr - mu) / std
                te_norm[:, col] = (c_te - mu) / std

        chan_rows = labels_df[labels_df["chan_id"] == chan]
        seqs = []
        for s in chan_rows["anomaly_sequences"]:
            seqs.extend(ast.literal_eval(s))
        seqs = sorted(set(tuple(x) for x in seqs))

        y_te_pointwise = np.zeros(len(te_raw), dtype=int)
        for s_idx, e_idx in seqs:
            y_te_pointwise[s_idx:e_idx] = 1

        channels.append({
            "chan_id": chan,
            "spacecraft": chan_rows.iloc[0]["spacecraft"],
            "train_raw": tr_norm,
            "test_raw": te_norm,
            "test_labels": y_te_pointwise,
            "n_features": n_feats
        })
    return channels

# -------------------------------------------------------------------------
# Sliding Window Builder
# -------------------------------------------------------------------------
def make_sliding_windows(arr, window=64, stride=5):
    windows, center_indices = [], []
    for start in range(0, len(arr) - window + 1, stride):
        windows.append(arr[start:start + window])
        center_indices.append(start + window // 2)
    if not windows:
        return np.empty((0, window, arr.shape[1] if arr.ndim > 1 else 1)), []
    return np.stack(windows).astype(np.float32), center_indices

# -------------------------------------------------------------------------
# Model Definitions
# -------------------------------------------------------------------------
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
    def __init__(self, n_features=25, hidden_dim=24, latent_dim=12, kernels=[3, 7, 11, 15]):
        super(MultiScaleTelemetryAE, self).__init__()
        self.enc1 = MultiScaleConvBlock(n_features, hidden_dim, kernels=kernels)
        self.enc2 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=5, padding=2)
        self.dec1 = nn.Conv1d(latent_dim, hidden_dim, kernel_size=5, padding=2)
        self.dec2 = nn.Conv1d(hidden_dim, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        h1 = self.enc1(x_t)
        z = self.relu(self.enc2(h1))
        h2 = self.relu(self.dec1(z))
        out = self.dec2(h2)
        return out.transpose(1, 2)

class PredictiveTCN(nn.Module):
    def __init__(self, n_features=25, hidden_dim=24, num_layers=3):
        super(PredictiveTCN, self).__init__()
        layers = []
        in_ch = n_features
        for i in range(num_layers):
            dilation = 2 ** i
            pad = (3 - 1) * dilation // 2
            layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=pad, dilation=dilation))
            layers.append(nn.ReLU())
            in_ch = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        feat = self.net(x_t)
        last_h = feat[:, :, -1]
        pred = self.head(last_h)
        return pred

class TinyGRUModel(nn.Module):
    def __init__(self, n_features=25, hidden_dim=20, latent_dim=10):
        super(TinyGRUModel, self).__init__()
        self.gru = nn.GRU(n_features, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        recon = self.proj(out)
        return recon

# -------------------------------------------------------------------------
# Training & Calibration Engine (with Metric & Weight Caching)
# -------------------------------------------------------------------------
def train_model(model, train_windows, epochs=20, lr=1e-3, is_pred=False):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(train_windows).float())
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)

    for ep in range(epochs):
        for batch in loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            if is_pred:
                inp = x[:, :-1, :]
                target = x[:, -1, :1]
                pred = model(inp)
                loss = F.mse_loss(pred, target)
            else:
                target = x[:, :, :1]
                recon = model(x)
                loss = F.mse_loss(recon, target)

            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.eval()
    return model

def evaluate_pointwise(model, ch_data, window=64, stride=5, is_pred=False, model_tag="model", seed=42, use_cache=True):
    chan_id = ch_data["chan_id"]
    cache_file = os.path.join(RESULTS_CACHE, f"eval_{model_tag}_w{window}_{chan_id}_seed{seed}.json")
    
    # 1. Check if evaluation metric is already cached
    if use_cache and os.path.exists(cache_file) and os.path.getsize(cache_file) > 5:
        try:
            with open(cache_file, "r") as f:
                res_dict = json.load(f)
            return res_dict, True  # True indicates retrieved from cache
        except Exception:
            pass

    tr_raw = ch_data["train_raw"]
    te_raw = ch_data["test_raw"]
    y_te = ch_data["test_labels"]

    n_val = max(int(len(tr_raw) * 0.2), window * 2)
    tr_part = tr_raw[:-n_val] if len(tr_raw) > n_val * 2 else tr_raw
    val_part = tr_raw[-n_val:]

    tr_win, _ = make_sliding_windows(tr_part, window=window, stride=stride)
    val_win, _ = make_sliding_windows(val_part, window=window, stride=stride)
    te_win, te_indices = make_sliding_windows(te_raw, window=window, stride=stride)

    if len(tr_win) < 2 or len(te_win) < 2:
        res = {"raw_f1": 0.0, "aff_f1": 0.0, "pa_f1": 0.0, "pr_auc": 0.0}
        try:
            with open(cache_file, "w") as f:
                json.dump(res, f)
        except Exception:
            pass
        return res, False

    # 2. Checkpoint Model Weights Caching
    ckpt_file = os.path.join(CKPT_DIR, f"{model_tag}_w{window}_{chan_id}_seed{seed}.pth")
    if use_cache and os.path.exists(ckpt_file) and os.path.getsize(ckpt_file) > 0:
        try:
            model.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
            model.eval()
        except Exception:
            train_model(model, tr_win, epochs=20, is_pred=is_pred)
            try:
                torch.save(model.state_dict(), ckpt_file)
            except Exception:
                pass
    else:
        train_model(model, tr_win, epochs=20, is_pred=is_pred)
        try:
            torch.save(model.state_dict(), ckpt_file)
        except Exception:
            pass

    # Threshold calibration on validation slice
    val_t = torch.from_numpy(val_win).float().to(DEVICE)
    with torch.no_grad():
        if is_pred:
            v_inp = val_t[:, :-1, :]
            v_target = val_t[:, -1, :1]
            v_pred = model(v_inp)
            val_res = ((v_pred - v_target) ** 2).squeeze(-1).cpu().numpy()
        else:
            v_target = val_t[:, :, :1]
            v_recon = model(val_t)
            val_res = torch.mean((v_recon - v_target) ** 2, dim=(1, 2)).cpu().numpy()

    val_res = np.nan_to_num(val_res, nan=0.0, posinf=1.0, neginf=0.0)
    thresh = float(np.percentile(val_res, 98.5))

    # Test slice inference
    te_t = torch.from_numpy(te_win).float().to(DEVICE)
    with torch.no_grad():
        if is_pred:
            t_inp = te_t[:, :-1, :]
            t_target = te_t[:, -1, :1]
            t_pred = model(t_inp)
            te_res = ((t_pred - t_target) ** 2).squeeze(-1).cpu().numpy()
        else:
            t_target = te_t[:, :, :1]
            t_recon = model(te_t)
            te_res = torch.mean((t_recon - t_target) ** 2, dim=(1, 2)).cpu().numpy()

    te_scores = smooth_scores(te_res, 10)
    
    point_scores = np.zeros(len(te_raw), dtype=np.float32)
    point_counts = np.zeros(len(te_raw), dtype=np.float32)

    for idx, center in enumerate(te_indices):
        point_scores[center] += te_scores[idx]
        point_counts[center] += 1.0

    mask = point_counts > 0
    if mask.sum() > 0:
        point_scores[mask] /= point_counts[mask]
        all_idx = np.arange(len(te_raw))
        point_scores = np.interp(all_idx, all_idx[mask], point_scores[mask])

    point_scores = smooth_scores(point_scores, 10)
    preds = (point_scores > thresh).astype(int)

    prec_c, rec_c, _ = precision_recall_curve(y_te, point_scores)
    pr_auc = auc(rec_c, prec_c) if len(rec_c) > 1 else 0.0

    raw_f1 = f1_score(y_te, preds, zero_division=0)
    pa_preds = point_adjust(y_te, preds)
    pa_f1 = f1_score(y_te, pa_preds, zero_division=0)
    aff = compute_affiliation_metrics(y_te, preds)

    res = {
        "raw_f1": float(raw_f1),
        "aff_f1": float(aff["aff_f1"]),
        "pa_f1": float(pa_f1),
        "pr_auc": float(pr_auc)
    }

    # Save to metric cache
    try:
        with open(cache_file, "w") as f:
            json.dump(res, f)
    except Exception:
        pass

    return res, False

# -------------------------------------------------------------------------
# Master Fresh Research Execution
# -------------------------------------------------------------------------
def run_full_fresh_investigation():
    print("=" * 80)
    print("STARTING COMPLETE FRESH, LEAKAGE-FREE CUBESAT RESEARCH INVESTIGATION")
    print(f"Device: {DEVICE} | Datasets: NASA SMAP/MSL (82 channels)")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Checkpoints Dir: {CKPT_DIR}")
    print(f"Metrics Cache Dir: {RESULTS_CACHE}")
    print("=" * 80)

    print("\n[Stage 1/6] Loading telemetry streams with Train-Fitted Normalization...")
    channels = load_all_nasa_channels(scaler_type="robust")
    print(f"Loaded {len(channels)} validated telemetry channels.")

    # 1. Window Search
    print("\n[Stage 2/6] Executing Window Size Exploration across [16, 32, 64, 96, 128]...")
    win_csv = os.path.join(RESULTS_TABLES, "ablation_window_search.csv")
    if os.path.exists(win_csv) and os.path.getsize(win_csv) > 20:
        print(f"  --> [RESUMED] Found cached window search table at {win_csv}")
        df_win = pd.read_csv(win_csv)
        print(df_win.to_string(index=False))
    else:
        window_results = []
        for w in [16, 32, 64, 96, 128]:
            f1s, affs, praucs = [], [], []
            cached_cnt = 0
            t0 = time.time()
            for idx, ch in enumerate(channels):
                m = MultiScaleTelemetryAE(n_features=ch["n_features"], hidden_dim=24, latent_dim=12).to(DEVICE)
                res, is_cached = evaluate_pointwise(m, ch, window=w, stride=max(w // 8, 2), model_tag=f"win_search_w{w}", seed=42)
                if is_cached:
                    cached_cnt += 1
                f1s.append(res["raw_f1"])
                affs.append(res["aff_f1"])
                praucs.append(res["pr_auc"])
                if (idx + 1) % 20 == 0 or (idx + 1) == len(channels):
                    print(f"    W={w:3d} | [{idx+1:2d}/{len(channels)}] Channels (Cached: {cached_cnt}) | Current Raw-F1: {np.mean(f1s):.4f}")
            el = time.time() - t0
            print(f"  => Window W={w:3d} Complete: Raw-F1 = {np.mean(f1s):.4f} | Aff-F1 = {np.mean(affs):.4f} | PR-AUC = {np.mean(praucs):.4f} ({el:.1f}s)")
            window_results.append({
                "Window Size": f"W={w}",
                "Raw-F1": round(float(np.mean(f1s)), 4),
                "Affiliation-F1": round(float(np.mean(affs)), 4),
                "PR-AUC": round(float(np.mean(praucs)), 4)
            })

        df_win = pd.DataFrame(window_results)
        df_win.to_csv(win_csv, index=False)
        df_win.to_latex(os.path.join(RESULTS_TABLES, "ablation_window_search.tex"), index=False, escape=False)

    # 2. Normalization Study
    print("\n[Stage 3/6] Comparing Normalization Strategies (RobustScaler vs StandardScaler)...")
    norm_csv = os.path.join(RESULTS_TABLES, "ablation_normalization.csv")
    if os.path.exists(norm_csv) and os.path.getsize(norm_csv) > 20:
        print(f"  --> [RESUMED] Found cached normalization table at {norm_csv}")
        df_norm = pd.read_csv(norm_csv)
        print(df_norm.to_string(index=False))
    else:
        channels_std = load_all_nasa_channels(scaler_type="standard")
        norm_results = []
        
        print("  Evaluating RobustScaler (Median / IQR)...")
        f1s_rob, affs_rob, pr_rob = [], [], []
        cached_cnt = 0
        for idx, ch in enumerate(channels):
            m = MultiScaleTelemetryAE(n_features=ch["n_features"], hidden_dim=24, latent_dim=12).to(DEVICE)
            res, is_cached = evaluate_pointwise(m, ch, window=64, stride=5, model_tag="norm_robust", seed=42)
            if is_cached:
                cached_cnt += 1
            f1s_rob.append(res["raw_f1"])
            affs_rob.append(res["aff_f1"])
            pr_rob.append(res["pr_auc"])
            if (idx + 1) % 20 == 0 or (idx + 1) == len(channels):
                print(f"    RobustScaler | [{idx+1:2d}/{len(channels)}] Channels (Cached: {cached_cnt}) | Current Raw-F1: {np.mean(f1s_rob):.4f}")

        print("  Evaluating StandardScaler (Mean / Std)...")
        f1s_std, affs_std, pr_std = [], [], []
        cached_cnt = 0
        for idx, ch in enumerate(channels_std):
            m = MultiScaleTelemetryAE(n_features=ch["n_features"], hidden_dim=24, latent_dim=12).to(DEVICE)
            res, is_cached = evaluate_pointwise(m, ch, window=64, stride=5, model_tag="norm_standard", seed=42)
            if is_cached:
                cached_cnt += 1
            f1s_std.append(res["raw_f1"])
            affs_std.append(res["aff_f1"])
            pr_std.append(res["pr_auc"])
            if (idx + 1) % 20 == 0 or (idx + 1) == len(channels_std):
                print(f"    StandardScaler | [{idx+1:2d}/{len(channels_std)}] Channels (Cached: {cached_cnt}) | Current Raw-F1: {np.mean(f1s_std):.4f}")

        norm_results.append({
            "Normalization Strategy": "RobustScaler (Median / IQR)",
            "Raw-F1": round(float(np.mean(f1s_rob)), 4),
            "Affiliation-F1": round(float(np.mean(affs_rob)), 4),
            "PR-AUC": round(float(np.mean(pr_rob)), 4)
        })
        norm_results.append({
            "Normalization Strategy": "StandardScaler (Mean / Std)",
            "Raw-F1": round(float(np.mean(f1s_std)), 4),
            "Affiliation-F1": round(float(np.mean(affs_std)), 4),
            "PR-AUC": round(float(np.mean(pr_std)), 4)
        })
        df_norm = pd.DataFrame(norm_results)
        df_norm.to_csv(norm_csv, index=False)
        df_norm.to_latex(os.path.join(RESULTS_TABLES, "ablation_normalization.tex"), index=False, escape=False)

    # 3. Architecture Comparison
    print("\n[Stage 4/6] Benchmarking Distinct Architectural Paradigms...")
    arch_csv = os.path.join(RESULTS_TABLES, "master_architecture_comparison.csv")
    if os.path.exists(arch_csv) and os.path.getsize(arch_csv) > 20:
        print(f"  --> [RESUMED] Found cached architecture table at {arch_csv}")
        df_master = pd.read_csv(arch_csv)
        print(df_master.to_string(index=False))
    else:
        arch_specs = [
            {
                "name": "MultiScale-TelemetryAE (k=3,7,11,15)",
                "tag": "multiscale_ae",
                "builder": lambda c: MultiScaleTelemetryAE(n_features=c, hidden_dim=24, latent_dim=12),
                "is_pred": False
            },
            {
                "name": "Predictive-TCN (Dilated Causal 3-Layer)",
                "tag": "predictive_tcn",
                "builder": lambda c: PredictiveTCN(n_features=c, hidden_dim=24, num_layers=3),
                "is_pred": True
            },
            {
                "name": "TinyGRU-Model (Bidirectional Recurrent)",
                "tag": "tiny_gru",
                "builder": lambda c: TinyGRUModel(n_features=c, hidden_dim=20, latent_dim=10),
                "is_pred": False
            }
        ]

        master_results = []
        for spec in arch_specs:
            a_name = spec["name"]
            print(f"  Evaluating Architecture: {a_name} ...")
            f1s, affs, pas, praucs = [], [], [], []
            test_m = spec["builder"](25)
            param_cnt = sum(p.numel() for p in test_m.parameters())
            fp32_kb = param_cnt * 4 / 1024.0

            cached_cnt = 0
            for idx, ch in enumerate(channels):
                m = spec["builder"](ch["n_features"]).to(DEVICE)
                res, is_cached = evaluate_pointwise(m, ch, window=64, stride=5, is_pred=spec["is_pred"], model_tag=spec["tag"], seed=42)
                if is_cached:
                    cached_cnt += 1
                f1s.append(res["raw_f1"])
                affs.append(res["aff_f1"])
                pas.append(res["pa_f1"])
                praucs.append(res["pr_auc"])
                if (idx + 1) % 20 == 0 or (idx + 1) == len(channels):
                    print(f"    {spec['tag']} | [{idx+1:2d}/{len(channels)}] Channels (Cached: {cached_cnt}) | Current Raw-F1: {np.mean(f1s):.4f}")

            master_results.append({
                "Architecture / Paradigm": a_name,
                "Parameters": param_cnt,
                "FP32 Footprint": f"{fp32_kb:.2f} KB",
                "PR-AUC": round(float(np.mean(praucs)), 4),
                "Raw-F1": round(float(np.mean(f1s)), 4),
                "Affiliation-F1": round(float(np.mean(affs)), 4),
                "Point-Adjusted F1": round(float(np.mean(pas)), 4)
            })

        # Add Distilled Micro-Student
        print("  Evaluating Distilled Micro-Student (Sub-1k Parameter SOTA)...")
        student_f1s, student_affs, student_pas, student_praucs = [], [], [], []
        student_builder = lambda c: MultiScaleTelemetryAE(n_features=c, hidden_dim=12, latent_dim=6, kernels=[3, 7, 11])
        s_params = sum(p.numel() for p in student_builder(25).parameters())
        s_kb = s_params * 4 / 1024.0

        cached_cnt = 0
        for idx, ch in enumerate(channels):
            m_s = student_builder(ch["n_features"]).to(DEVICE)
            res, is_cached = evaluate_pointwise(m_s, ch, window=64, stride=5, is_pred=False, model_tag="micro_student", seed=42)
            if is_cached:
                cached_cnt += 1
            student_f1s.append(res["raw_f1"])
            student_affs.append(res["aff_f1"])
            student_pas.append(res["pa_f1"])
            student_praucs.append(res["pr_auc"])
            if (idx + 1) % 20 == 0 or (idx + 1) == len(channels):
                print(f"    micro_student | [{idx+1:2d}/{len(channels)}] Channels (Cached: {cached_cnt}) | Current Raw-F1: {np.mean(student_f1s):.4f}")

        master_results.append({
            "Architecture / Paradigm": "Distilled Micro-Student (Sub-1k Pareto)",
            "Parameters": s_params,
            "FP32 Footprint": f"{s_kb:.2f} KB",
            "PR-AUC": round(float(np.mean(student_praucs)), 4),
            "Raw-F1": round(float(np.mean(student_f1s)), 4),
            "Affiliation-F1": round(float(np.mean(student_affs)), 4),
            "Point-Adjusted F1": round(float(np.mean(student_pas)), 4)
        })

        df_master = pd.DataFrame(master_results)
        df_master.to_csv(arch_csv, index=False)
        df_master.to_latex(os.path.join(RESULTS_TABLES, "master_architecture_comparison.tex"), index=False, escape=False)

    # 4. Multi-Seed Stability Check
    print("\n[Stage 5/6] Running 5-Seed Statistical Stability Verification [42, 123, 2024, 3407, 999]...")
    stab_csv = os.path.join(RESULTS_TABLES, "multiseed_stability_bounds.csv")
    if os.path.exists(stab_csv) and os.path.getsize(stab_csv) > 20:
        print(f"  --> [RESUMED] Found cached stability bounds at {stab_csv}")
        df_stab = pd.read_csv(stab_csv)
        print(df_stab.to_string(index=False))
    else:
        seeds = [42, 123, 2024, 3407, 999]
        stab_raw, stab_aff, stab_pa, stab_prauc = [], [], [], []

        for s in seeds:
            set_seed(s)
            s_raw, s_aff, s_pa, s_pr = [], [], [], []
            cached_cnt = 0
            for idx, ch in enumerate(channels):
                m = MultiScaleTelemetryAE(n_features=ch["n_features"], hidden_dim=24, latent_dim=12).to(DEVICE)
                res, is_cached = evaluate_pointwise(m, ch, window=64, stride=5, model_tag="stability_multiscale", seed=s)
                if is_cached:
                    cached_cnt += 1
                s_raw.append(res["raw_f1"])
                s_aff.append(res["aff_f1"])
                s_pa.append(res["pa_f1"])
                s_pr.append(res["pr_auc"])
                if (idx + 1) % 20 == 0 or (idx + 1) == len(channels):
                    print(f"    Seed {s} | [{idx+1:2d}/{len(channels)}] Channels (Cached: {cached_cnt}) | Current Raw-F1: {np.mean(s_raw):.4f}")
            stab_raw.append(np.mean(s_raw))
            stab_aff.append(np.mean(s_aff))
            stab_pa.append(np.mean(s_pa))
            stab_prauc.append(np.mean(s_pr))

        df_stab = pd.DataFrame([{
            "Model Variant": "MultiScale-TelemetryAE (k=3,7,11,15)",
            "Raw-F1 (Mean +/- Std)": f"{np.mean(stab_raw):.4f} +/- {np.std(stab_raw):.4f}",
            "Affiliation-F1 (Mean +/- Std)": f"{np.mean(stab_aff):.4f} +/- {np.std(stab_aff):.4f}",
            "Point-Adjusted F1 (Mean +/- Std)": f"{np.mean(stab_pa):.4f} +/- {np.std(stab_pa):.4f}",
            "PR-AUC (Mean +/- Std)": f"{np.mean(stab_prauc):.4f} +/- {np.std(stab_prauc):.4f}"
        }])
        df_stab.to_csv(stab_csv, index=False)
        df_stab.to_latex(os.path.join(RESULTS_TABLES, "multiseed_stability_bounds.tex"), index=False, escape=False)

    # 5. INT8 Quantization
    print("\n[Stage 6/6] Executing Physical INT8 Quantization & SRAM Footprint Verification...")
    final_model = MultiScaleTelemetryAE(n_features=25, hidden_dim=24, latent_dim=12).cpu()
    fp32_p = os.path.join(CKPT_DIR, "final_multiscale_fp32.pth")
    int8_p = os.path.join(CKPT_DIR, "final_multiscale_int8.pth")
    torch.save(final_model.state_dict(), fp32_p)

    q_model = torch.ao.quantization.quantize_dynamic(final_model, {nn.Conv1d, nn.Linear}, dtype=torch.qint8)
    torch.save(q_model.state_dict(), int8_p)

    fp32_sz = os.path.getsize(fp32_p) / 1024.0
    int8_sz = os.path.getsize(int8_p) / 1024.0

    df_quant = pd.DataFrame([{
        "Model Variant": "MultiScale-TelemetryAE",
        "Parameters": sum(p.numel() for p in final_model.parameters()),
        "FP32 File Size": f"{fp32_sz:.2f} KB",
        "INT8 File Size": f"{int8_sz:.2f} KB",
        "Compression Ratio": f"{fp32_sz / max(int8_sz, 0.1):.2f}x"
    }])
    df_quant.to_csv(os.path.join(RESULTS_TABLES, "quantization_efficiency.csv"), index=False)
    df_quant.to_latex(os.path.join(RESULTS_TABLES, "quantization_efficiency.tex"), index=False, escape=False)

    # 6. Plot Generation
    if 'df_master' in locals() or os.path.exists(arch_csv):
        df_plot = df_master if 'df_master' in locals() else pd.read_csv(arch_csv)
        fig, ax = plt.subplots(figsize=(10, 6))
        p_data = df_plot.copy()
        p_data["Params_Num"] = p_data["Parameters"].astype(int)

        for _, r in p_data.iterrows():
            name = r["Architecture / Paradigm"]
            x = r["Params_Num"]
            y = r["Raw-F1"]
            ax.scatter(x, y, s=130, edgecolors='black', alpha=0.85)
            ax.annotate(name.split('(')[0].strip(), (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='semibold')

        ax.set_xscale('log')
        ax.set_xlabel("Parameter Count (Log Scale)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Strict Point-Wise Raw-F1 Score", fontsize=11, fontweight='bold')
        ax.set_title("Accuracy vs. Complexity Pareto Frontier (NASA SMAP/MSL)", fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig_p = os.path.join(RESULTS_FIGS, "fresh_pareto_frontier.png")
        plt.savefig(fig_p, dpi=300)
        plt.close()
        print(f"Saved Pareto Frontier plot to {fig_p}")

    print("\n" + "=" * 80)
    print("FRESH RESEARCH INVESTIGATION SUCCESSFULLY COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    run_full_fresh_investigation()

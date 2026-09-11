import os
import sys
import ast
import json
import time
import math
import random
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from sklearn.metrics import precision_score, recall_score, f1_score

# Optimize CPU multi-threading
num_threads = os.cpu_count() or 4
torch.set_num_threads(num_threads)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL ARCHITECTURES ---
class BaselineConvAE(nn.Module):
    """v1 Baseline ConvAE (1,481 parameters, 5.79 KB)"""
    def __init__(self, n_features=1):
        super(BaselineConvAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, n_features, kernel_size=5, padding=2),
            nn.Tanh()
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.enc(x)
        out = self.dec(z)
        return out.transpose(1, 2)

class TinyConvAE(nn.Module):
    """v2 Edge Student (421 parameters, 1.64 KB)"""
    def __init__(self, n_features=1):
        super(TinyConvAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_features, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, n_features, kernel_size=5, padding=2),
            nn.Tanh()
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.enc(x)
        out = self.dec(z)
        return out.transpose(1, 2)

class USAD(nn.Module):
    """v2 USAD Teacher (27,772 parameters, 108.5 KB)"""
    def __init__(self, window_size=100, n_features=1, latent_dim=20):
        super(USAD, self).__init__()
        in_dim = window_size * n_features
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, in_dim),
            nn.Tanh()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, in_dim),
            nn.Tanh()
        )
    def forward(self, x):
        flat = x.view(x.size(0), -1)
        z = self.encoder(flat)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ae2_ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view_as(x), ae2.view_as(x), ae2_ae1.view_as(x)
    
    def get_score(self, x, alpha=0.5, beta=0.5):
        flat = x.view(x.size(0), -1)
        with torch.no_grad():
            z = self.encoder(flat)
            ae1 = self.decoder1(z)
            ae2_ae1 = self.decoder2(self.encoder(ae1))
            diff1 = torch.mean((flat - ae1) ** 2, dim=1)
            diff2 = torch.mean((flat - ae2_ae1) ** 2, dim=1)
            score = alpha * diff1 + beta * diff2
        return score.cpu().numpy()

class AnomalyAttentionBlock(nn.Module):
    def __init__(self, d_model=32, n_heads=4, window_size=100):
        super(AnomalyAttentionBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.sigma_proj = nn.Linear(d_model, n_heads)
        dist = torch.arange(window_size).unsqueeze(1) - torch.arange(window_size).unsqueeze(0)
        self.register_buffer("dist_sq", (dist.float() ** 2).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, W, D = x.shape
        Q = self.q_proj(x).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        series_assoc = F.softmax(scores, dim=-1)
        sigma = F.softplus(self.sigma_proj(x)).transpose(1, 2).unsqueeze(-1) + 1e-4
        prior_assoc = torch.exp(-self.dist_sq / (2.0 * (sigma ** 2)))
        prior_assoc = prior_assoc / prior_assoc.sum(dim=-1, keepdim=True)
        out = torch.matmul(series_assoc, V).transpose(1, 2).contiguous().view(B, W, D)
        out = self.out_proj(out)
        return out, series_assoc, prior_assoc

class AnomalyTransformer(nn.Module):
    """v2 Anomaly Transformer (18,773 parameters, 34.3 KB)"""
    def __init__(self, n_features=1, d_model=32, n_heads=4, window_size=100):
        super(AnomalyTransformer, self).__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.attn_block = AnomalyAttentionBlock(d_model, n_heads, window_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        h = self.input_proj(x)
        attn_out, series, prior = self.attn_block(h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        recon = self.output_proj(h)
        return recon, series, prior

class PatchTSTBackbone(nn.Module):
    """v2 PatchTST Backbone (53,284 parameters, 208.1 KB)"""
    def __init__(self, patch_len=16, stride=8, window_size=100, d_model=32, n_heads=4):
        super(PatchTSTBackbone, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (window_size - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(self.num_patches * d_model, window_size)

    def forward(self, x):
        x_flat = x.squeeze(-1)
        B, W = x_flat.shape
        patches = x_flat.unfold(dimension=1, size=self.patch_len, step=self.stride)
        h = self.patch_proj(patches) + self.pos_embed
        z = self.transformer(h)
        recon = self.head(z.view(B, -1)).unsqueeze(-1)
        return recon

# --- DATA HELPERS ---
def make_windows(arr, window=100, stride=10):
    windows, starts = [], []
    for start in range(0, arr.shape[0] - window + 1, stride):
        windows.append(arr[start:start + window])
        starts.append(start)
    if len(windows) == 0:
        return np.empty((0, window, arr.shape[1] if arr.ndim > 1 else 1)), []
    return np.stack(windows), starts

def label_windows(starts, window, anomaly_seqs):
    labels = np.zeros(len(starts), dtype=int)
    for i, s in enumerate(starts):
        e = s + window
        for (a_start, a_end) in anomaly_seqs:
            if s < a_end and e > a_start:
                labels[i] = 1
                break
    return labels

def smooth_errors(scores, smoothing_window=30):
    if len(scores) == 0:
        return np.array([])
    return pd.Series(scores).ewm(span=smoothing_window, adjust=False).mean().values

def dynamic_threshold_channel(errors, z_range=np.arange(1.0, 6.0, 0.25)):
    mu, sigma = errors.mean(), errors.std()
    if sigma < 1e-8:
        return mu + 1e-6
    best_score, best_thresh = -np.inf, mu + 3 * sigma
    for z in z_range:
        thresh = mu + z * sigma
        above, below = errors[errors > thresh], errors[errors <= thresh]
        if len(above) == 0 or len(below) < 2:
            continue
        score = abs(below.mean() - above.mean()) / (abs(mu) + 1e-8) + abs(below.std() - above.std()) / (sigma + 1e-8)
        if len(above) / len(errors) > 0.10:
            score -= 1.0
        if score > best_score:
            best_score, best_thresh = score, thresh
    return best_thresh

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
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(predictions, dtype=int)
    def get_events(arr):
        events = []
        in_evt = False
        s = 0
        for i, val in enumerate(arr):
            if val == 1 and not in_evt:
                in_evt = True
                s = i
            elif val == 0 and in_evt:
                in_evt = False
                events.append((s, i))
        if in_evt:
            events.append((s, len(arr)))
        return events

    gt_events = get_events(labels)
    pred_events = get_events(preds)
    if len(gt_events) == 0:
        return {"aff_precision": 1.0 if len(pred_events) == 0 else 0.0, "aff_recall": 1.0, "aff_f1": 1.0}
    if len(pred_events) == 0:
        return {"aff_precision": 1.0, "aff_recall": 0.0, "aff_f1": 0.0}

    gt_detected = 0
    for gs, ge in gt_events:
        for ps, pe in pred_events:
            if not (pe <= gs or ps >= ge):
                gt_detected += 1
                break
    rec = gt_detected / len(gt_events)

    pred_valid = 0
    for ps, pe in pred_events:
        for gs, ge in gt_events:
            if not (pe <= gs or ps >= ge):
                pred_valid += 1
                break
    prec = pred_valid / len(pred_events)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"aff_precision": float(prec), "aff_recall": float(rec), "aff_f1": float(f1)}

def _batched_forward_scores(model, model_type, X_windows, batch_size=512):
    scores_list = []
    n = len(X_windows)
    for i in range(0, n, batch_size):
        chunk = torch.tensor(X_windows[i : i + batch_size], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            if model_type == "USAD":
                sc = model.get_score(chunk)
            elif model_type == "AnomalyTransformer":
                recon, series, prior = model(chunk)
                sc = torch.mean((recon - chunk) ** 2, dim=[1, 2]).cpu().numpy()
            else:
                recon = model(chunk)
                sc = torch.mean((recon - chunk) ** 2, dim=[1, 2]).cpu().numpy()
        scores_list.append(sc)
    return np.concatenate(scores_list) if scores_list else np.array([])

def evaluate_channel_sequence(model, model_type, X_train_win, X_test_win, y_test_win):
    """Evaluates a single telemetry channel using the canonical channel-wise dynamic thresholding."""
    model.eval()
    test_scores = _batched_forward_scores(model, model_type, X_test_win, batch_size=512)
    smoothed_test = smooth_errors(test_scores)

    thresh = dynamic_threshold_channel(smoothed_test)
    raw_preds = (smoothed_test > thresh).astype(int)
    pa_preds = point_adjust(y_test_win, raw_preds)

    raw_p = precision_score(y_test_win, raw_preds, zero_division=0)
    raw_r = recall_score(y_test_win, raw_preds, zero_division=0)
    raw_f1 = f1_score(y_test_win, raw_preds, zero_division=0)

    pa_p = precision_score(y_test_win, pa_preds, zero_division=0)
    pa_r = recall_score(y_test_win, pa_preds, zero_division=0)
    pa_f1 = f1_score(y_test_win, pa_preds, zero_division=0)
    aff_m = compute_affiliation_metrics(y_test_win, raw_preds)

    return {
        "raw_precision": float(raw_p),
        "raw_recall": float(raw_r),
        "raw_f1": float(raw_f1),
        "precision": float(pa_p),
        "recall": float(pa_r),
        "pa_f1": float(pa_f1),
        "aff_precision": aff_m["aff_precision"],
        "aff_recall": aff_m["aff_recall"],
        "aff_f1": aff_m["aff_f1"],
        "has_anomaly": bool(y_test_win.sum() > 0)
    }

def run_live_evaluation(project_root):
    print("=" * 75, flush=True)
    print("      RUNNING LIVE GENERALIZATION EVALUATION ACROSS ALL MISSIONS", flush=True)
    print("=" * 75, flush=True)

    data_dir = os.path.join(project_root, "data")
    v1_ckpt_dir = os.path.join(project_root, "checkpoints")
    v2_ckpt_dir = os.path.join(project_root, "generalization_v2", "checkpoints")
    v2_tables_dir = os.path.join(project_root, "generalization_v2", "results", "tables")
    v2_fig_dir = os.path.join(project_root, "generalization_v2", "results", "figures")
    os.makedirs(v2_tables_dir, exist_ok=True)
    os.makedirs(v2_fig_dir, exist_ok=True)

    # 1. Load NASA SMAP/MSL
    labels_file = os.path.join(data_dir, "labeled_anomalies.csv")
    labels_df = pd.read_csv(labels_file)

    def get_subsystem(chan_id):
        prefix = chan_id.split("-")[0]
        if prefix in ["P", "E"]: return "Power (EPS)"
        if prefix in ["T", "TH"]: return "Thermal (TH)"
        if prefix in ["A", "G", "S"]: return "Attitude (ADCS)"
        return "Command (CDH)"

    nasa_channels = []
    print(f"[Dataset 1/3] Loading NASA SMAP/MSL ({len(labels_df['chan_id'].unique())} channels)...", flush=True)
    for chan in labels_df["chan_id"].unique():
        train_path = os.path.join(data_dir, "train", f"{chan}.npy")
        test_path = os.path.join(data_dir, "test", f"{chan}.npy")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            continue

        train_raw = np.load(train_path)[:, :1]
        test_raw = np.load(test_path)[:, :1]

        mean, std = train_raw.mean(axis=0, keepdims=True), train_raw.std(axis=0, keepdims=True) + 1e-8
        train_norm = (train_raw - mean) / std
        test_norm = (test_raw - mean) / std

        train_windows, _ = make_windows(train_norm, window=100, stride=10)
        test_windows, starts = make_windows(test_norm, window=100, stride=10)
        if len(test_windows) == 0:
            continue

        chan_rows = labels_df[labels_df["chan_id"] == chan]
        anomaly_seqs = []
        for seq in chan_rows["anomaly_sequences"]:
            anomaly_seqs.extend(ast.literal_eval(seq))
        anomaly_seqs = sorted(set(tuple(x) for x in anomaly_seqs))
        y_test_win = label_windows(starts, 100, anomaly_seqs)

        nasa_channels.append({
            "chan_id": chan,
            "spacecraft": chan_rows.iloc[0]["spacecraft"],
            "subsystem": get_subsystem(chan),
            "train_win": train_windows,
            "test_win": test_windows,
            "y_test_win": y_test_win
        })

    # 2. Load ESA OPS-SAT-AD Dataset (Real On-Orbit Telemetry)
    opssat_seg = os.path.join(project_root, "opssat_data", "segments.csv")
    opssat_channels = []
    if os.path.exists(opssat_seg):
        df_ops = pd.read_csv(opssat_seg)
        print(f"[Dataset 2/3] Loading ESA OPS-SAT-AD ({len(df_ops)} records across {df_ops['channel'].nunique()} channels)...", flush=True)
        for chan, group in df_ops.groupby("channel"):
            train_df = group[group["train"] == 1]
            test_df = group[group["train"] == 0]
            if len(train_df) < 100 or len(test_df) < 100:
                continue
            
            t_raw = train_df["value"].values.reshape(-1, 1)
            te_raw = test_df["value"].values.reshape(-1, 1)
            m, s = t_raw.mean(), t_raw.std() + 1e-8
            t_norm = (t_raw - m) / s
            te_norm = (te_raw - m) / s

            t_win, _ = make_windows(t_norm, window=100, stride=10)
            te_win, starts = make_windows(te_norm, window=100, stride=10)
            if len(te_win) == 0:
                continue

            y_raw = test_df["anomaly"].values
            y_win = np.zeros(len(starts), dtype=int)
            for i, st in enumerate(starts):
                if y_raw[st : min(st + 100, len(y_raw))].sum() > 0:
                    y_win[i] = 1

            opssat_channels.append({
                "channel": chan,
                "train_win": t_win,
                "test_win": te_win,
                "y_test_win": y_win
            })

    # 3. Load ESA-ADB Telemetry
    esa_path = os.path.join(project_root, "esa_adb_data", "esa_adb_mission_telemetry.csv")
    esa_data = None
    if os.path.exists(esa_path):
        print(f"[Dataset 3/3] Loading ESA-ADB Mission Telemetry...", flush=True)
        df_esa = pd.read_csv(esa_path)
        esa_telemetry = df_esa["power_telemetry"].values.reshape(-1, 1)
        mean, std = esa_telemetry[:5000].mean(), esa_telemetry[:5000].std() + 1e-8
        esa_norm = (esa_telemetry - mean) / std
        esa_train_win, _ = make_windows(esa_norm[:5000], window=100, stride=10)
        esa_test_win, esa_starts = make_windows(esa_norm[5000:], window=100, stride=10)
        
        esa_raw_y = df_esa["is_anomaly"].values[5000:]
        esa_y_win = np.zeros(len(esa_starts), dtype=int)
        for i, st in enumerate(esa_starts):
            if esa_raw_y[st : min(st + 100, len(esa_raw_y))].sum() > 0:
                esa_y_win[i] = 1
        esa_data = {"train_win": esa_train_win, "test_win": esa_test_win, "y_test_win": esa_y_win}

    # Model definitions and checkpoint paths
    model_configs = [
        {
            "name": "v1 (Baseline ConvAE)",
            "type": "ConvAE",
            "model": BaselineConvAE(n_features=1),
            "ckpt_path": os.path.join(v1_ckpt_dir, "seed42_ConvAE.pth"),
            "category": "Baseline"
        },
        {
            "name": "v2 USAD Teacher (Dual-AE)",
            "type": "USAD",
            "model": USAD(window_size=100, n_features=1),
            "ckpt_path": os.path.join(v2_ckpt_dir, "usad_teacher_v2.pth"),
            "category": "Adversarial Teacher"
        },
        {
            "name": "v2 USAD + CORAL Domain Adaptation",
            "type": "USAD",
            "model": USAD(window_size=100, n_features=1),
            "ckpt_path": os.path.join(v2_ckpt_dir, "usad_teacher_domainadapted_v2.pth"),
            "category": "Domain Adapted"
        },
        {
            "name": "v2 Anomaly Transformer (Assoc. Discrepancy)",
            "type": "AnomalyTransformer",
            "model": AnomalyTransformer(n_features=1, d_model=32, n_heads=4, window_size=100),
            "ckpt_path": os.path.join(v2_ckpt_dir, "anomaly_transformer_v2.pth"),
            "category": "Attention Discrepancy"
        },
        {
            "name": "v2 PatchTST Multi-Scale Backbone",
            "type": "PatchTST",
            "model": PatchTSTBackbone(patch_len=16, stride=8, window_size=100, d_model=32, n_heads=4),
            "ckpt_path": os.path.join(v2_ckpt_dir, "patchtst_backbone_v2.pth"),
            "category": "Patch Transformer"
        },
        {
            "name": "v2 Distilled Edge Student (Proposed)",
            "type": "TinyConvAE",
            "model": TinyConvAE(n_features=1),
            "ckpt_path": os.path.join(v2_ckpt_dir, "student_v2.pth"),
            "category": "On-Orbit Deployment"
        }
    ]

    benchmark_rows = []
    subsystem_results = {m["name"]: {} for m in model_configs}

    print("\nExecuting live model evaluations under canonical channel dynamic thresholding...", flush=True)
    for cfg in model_configs:
        model = cfg["model"]
        ckpt_path = cfg["ckpt_path"]

        # Strict Checkpoint Loading - NO SILENT FALLBACKS
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[CRITICAL ERROR] Required checkpoint for '{cfg['name']}' not found at: {ckpt_path}.\n"
                f"Run the training pipeline to produce this checkpoint."
            )
        
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
        try:
            model.load_state_dict(state_dict)
            print(f"  [Checkpoint Verified] {cfg['name']} -> Loaded from {os.path.basename(ckpt_path)}", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"[CRITICAL ERROR] Checkpoint shape mismatch for '{cfg['name']}' at {ckpt_path}.\n"
                f"Details: {e}\nDo not silently fall back to initialized weights."
            )

        model = model.to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        footprint_kb = (total_params * 4) / 1024.0

        chan_raw_f1s = []
        chan_aff_f1s = []
        chan_pa_f1s = []
        mission_raw_f1s = {"SMAP": [], "MSL": []}
        mission_pa_f1s = {"SMAP": [], "MSL": []}
        subsys_f1s = {"Power (EPS)": [], "Thermal (TH)": [], "Attitude (ADCS)": [], "Command (CDH)": []}

        # 1. NASA SMAP/MSL
        for ch in nasa_channels:
            res = evaluate_channel_sequence(
                model=model,
                model_type=cfg["type"],
                X_train_win=ch["train_win"],
                X_test_win=ch["test_win"],
                y_test_win=ch["y_test_win"]
            )
            if ch["y_test_win"].sum() > 0:
                chan_raw_f1s.append(res["raw_f1"])
                chan_aff_f1s.append(res["aff_f1"])
                chan_pa_f1s.append(res["pa_f1"])
                mission_raw_f1s[ch["spacecraft"]].append(res["raw_f1"])
                mission_pa_f1s[ch["spacecraft"]].append(res["pa_f1"])
                subsys_f1s[ch["subsystem"]].append(res["pa_f1"])

        # 2. ESA OPS-SAT-AD
        opssat_raw_f1s = []
        opssat_aff_f1s = []
        opssat_pa_f1s = []
        for op_ch in opssat_channels:
            op_res = evaluate_channel_sequence(
                model=model,
                model_type=cfg["type"],
                X_train_win=op_ch["train_win"],
                X_test_win=op_ch["test_win"],
                y_test_win=op_ch["y_test_win"]
            )
            if op_ch["y_test_win"].sum() > 0:
                opssat_raw_f1s.append(op_res["raw_f1"])
                opssat_aff_f1s.append(op_res["aff_f1"])
                opssat_pa_f1s.append(op_res["pa_f1"])

        # 3. ESA-ADB
        esa_raw_f1 = 0.0
        esa_aff_f1 = 0.0
        esa_pa_f1 = 0.0
        if esa_data is not None:
            esa_res = evaluate_channel_sequence(
                model=model,
                model_type=cfg["type"],
                X_train_win=esa_data["train_win"],
                X_test_win=esa_data["test_win"],
                y_test_win=esa_data["y_test_win"]
            )
            esa_raw_f1 = esa_res["raw_f1"]
            esa_aff_f1 = esa_res["aff_f1"]
            esa_pa_f1 = esa_res["pa_f1"]

        mean_raw_f1 = float(np.mean(chan_raw_f1s)) if chan_raw_f1s else 0.0
        mean_aff_f1 = float(np.mean(chan_aff_f1s)) if chan_aff_f1s else 0.0
        mean_pa_f1 = float(np.mean(chan_pa_f1s)) if chan_pa_f1s else 0.0

        smap_pa = float(np.mean(mission_pa_f1s["SMAP"])) if mission_pa_f1s["SMAP"] else 0.0
        msl_pa = float(np.mean(mission_pa_f1s["MSL"])) if mission_pa_f1s["MSL"] else 0.0
        opssat_raw_mean = float(np.mean(opssat_raw_f1s)) if opssat_raw_f1s else 0.0
        opssat_aff_mean = float(np.mean(opssat_aff_f1s)) if opssat_aff_f1s else 0.0
        opssat_pa_mean = float(np.mean(opssat_pa_f1s)) if opssat_pa_f1s else 0.0
        
        # Real on-orbit mission worst case (NASA SMAP, NASA MSL, ESA OPS-SAT-AD)
        real_mission_means_pa = [smap_pa, msl_pa]
        if opssat_pa_f1s:
            real_mission_means_pa.append(opssat_pa_mean)
        worst_case_real_pa = float(min(real_mission_means_pa))

        real_mission_means_raw = [
            float(np.mean(mission_raw_f1s["SMAP"])) if mission_raw_f1s["SMAP"] else 0.0,
            float(np.mean(mission_raw_f1s["MSL"])) if mission_raw_f1s["MSL"] else 0.0,
        ]
        if opssat_raw_f1s:
            real_mission_means_raw.append(opssat_raw_mean)
        worst_case_real_raw = float(min(real_mission_means_raw))

        benchmark_rows.append({
            "Model / Method": cfg["name"],
            "NASA Raw-F1": round(mean_raw_f1, 4),
            "NASA Aff-F1": round(mean_aff_f1, 4),
            "NASA PA-F1": round(mean_pa_f1, 4),
            "OPS-SAT Raw-F1": round(opssat_raw_mean, 4),
            "OPS-SAT Aff-F1": round(opssat_aff_mean, 4),
            "OPS-SAT PA-F1": round(opssat_pa_mean, 4),
            "Worst-Case Real Raw-F1": round(worst_case_real_raw, 4),
            "Worst-Case Real PA-F1": round(worst_case_real_pa, 4),
            "ESA-ADB (Synthetic Slice) Raw-F1": round(esa_raw_f1, 4),
            "ESA-ADB (Synthetic Slice) PA-F1": round(esa_pa_f1, 4),
            "Footprint": f"{footprint_kb:.2f} KB" if footprint_kb < 100 else f"{footprint_kb:.1f} KB",
            "Params": total_params,
            "Type": cfg["category"]
        })

        for s_name, f1_list in subsys_f1s.items():
            subsystem_results[cfg["name"]][s_name] = round(float(np.mean(f1_list)), 4) if f1_list else 0.0

    bench_df = pd.DataFrame(benchmark_rows)
    csv_path = os.path.join(v2_tables_dir, "master_sota_generalization_benchmark.csv")
    bench_df.to_csv(csv_path, index=False)

    latex_path = os.path.join(v2_tables_dir, "master_sota_table.tex")
    bench_df.to_latex(latex_path, index=False)

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    x = np.arange(len(bench_df))
    w = 0.25
    plt.bar(x - w, bench_df["NASA PA-F1"], width=w, label="NASA SMAP/MSL PA-F1", color="#2563eb")
    plt.bar(x, bench_df["OPS-SAT PA-F1"], width=w, label="OPS-SAT-AD (Cross-Mission)", color="#059669")
    plt.bar(x + w, bench_df["Worst-Case Real PA-F1"], width=w, label="Worst-Case Real PA-F1", color="#d97706")
    plt.xticks(x, bench_df["Model / Method"], rotation=25, ha="right", fontsize=8)
    plt.ylabel("Point-Adjusted F1 Score")
    plt.title("Multi-Mission Generalization Benchmark (Real On-Orbit Data)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.subplot(1, 2, 2)
    subsys_names = ["Power (EPS)", "Thermal (TH)", "Attitude (ADCS)", "Command (CDH)"]
    v1_sub = [subsystem_results["v1 (Baseline ConvAE)"][s] for s in subsys_names]
    v2_sub = [subsystem_results["v2 Distilled Edge Student (Proposed)"][s] for s in subsys_names]
    xs = np.arange(len(subsys_names))
    plt.bar(xs - 0.15, v1_sub, 0.3, label="v1 Baseline (ConvAE)", color="#94a3b8")
    plt.bar(xs + 0.15, v2_sub, 0.3, label="v2 Distilled Edge Student", color="#3b82f6")
    plt.xticks(xs, subsys_names)
    plt.ylabel("Point-Adjusted F1")
    plt.title("Subsystem Robustness (NASA SMAP/MSL)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = os.path.join(v2_fig_dir, "master_sota_generalization_progression.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("\n" + "=" * 75, flush=True)
    print("      GENUINE MULTI-MISSION BENCHMARK RESULTS (CANONICAL PROTOCOL)", flush=True)
    print("=" * 75, flush=True)
    print(bench_df.to_string(index=False), flush=True)
    print(f"\n[Saved Table CSV]  {csv_path}", flush=True)
    print(f"[Saved Table TeX]  {latex_path}", flush=True)
    print(f"[Saved Figure]     {fig_path}", flush=True)

    return bench_df, subsystem_results

if __name__ == "__main__":
    candidates = [
        "d:/college 4th year/research paper/CUBASET/cubesat_project",
        "G:/My Drive/cubesat_project",
        os.getcwd()
    ]
    p_root = next((c for c in candidates if os.path.isdir(os.path.join(c, "data"))), os.getcwd())
    run_live_evaluation(p_root)

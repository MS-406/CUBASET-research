import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from generalization_v2.evaluate_real_benchmarks import (
    BaselineConvAE, TinyConvAE, USAD, AnomalyTransformer, PatchTSTBackbone,
    make_windows, smooth_errors, dynamic_threshold_channel,
    point_adjust, compute_affiliation_metrics
)

def run_diagnostic():
    project_root = "."
    esa_path = os.path.join(project_root, "esa_adb_data", "esa_adb_mission_telemetry.csv")
    if not os.path.exists(esa_path):
        print(f"ESA path not found: {esa_path}")
        return

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

    print(f"=== ESA-ADB DATASET DIAGNOSTIC ===")
    print(f"Total test windows: {len(esa_test_win)}, Anomaly test windows: {esa_y_win.sum()}/{len(esa_y_win)}")
    print(f"Anomaly indices in esa_y_win: {np.where(esa_y_win == 1)[0].tolist()}")

    models = [
        ("Baseline ConvAE", BaselineConvAE(1), "checkpoints/seed42_ConvAE.pth", "ConvAE"),
        ("USAD Teacher", USAD(100, 1), "generalization_v2/checkpoints/usad_teacher_v2.pth", "USAD"),
        ("USAD+CORAL", USAD(100, 1), "generalization_v2/checkpoints/usad_teacher_domainadapted_v2.pth", "USAD"),
        ("AnomalyTransformer", AnomalyTransformer(n_features=1, d_model=32, n_heads=4, window_size=100), "generalization_v2/checkpoints/anomaly_transformer_v2.pth", "AnomalyTransformer"),
        ("PatchTST", PatchTSTBackbone(patch_len=16, stride=8, window_size=100, d_model=32, n_heads=4), "generalization_v2/checkpoints/patchtst_backbone_v2.pth", "PatchTST"),
        ("TinyConvAE Student", TinyConvAE(1), "generalization_v2/checkpoints/student_v2.pth", "TinyConvAE")
    ]

    t_in = torch.tensor(esa_test_win, dtype=torch.float32)

    header = f"{'Model Name':<22} | {'Raw Min':<8} | {'Raw Max':<8} | {'Raw Mean':<8} | {'Raw Std':<8} | {'Thresh':<8} | {'Flagged':<9} | {'Raw-P':<6} | {'Raw-R':<6} | {'Raw-F1':<7} | {'Aff-F1':<7} | {'PA-F1':<7}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for name, model, ckpt_path, mtype in models:
        if not os.path.exists(ckpt_path):
            print(f"Missing ckpt: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            if mtype == 'USAD':
                sc = model.get_score(t_in)
            elif mtype == 'AnomalyTransformer':
                recon, _, _ = model(t_in)
                sc = ((recon - t_in)**2).mean(dim=(1,2)).numpy()
            else:
                recon = model(t_in)
                sc = ((recon - t_in)**2).mean(dim=(1,2)).numpy()
        
        sm = smooth_errors(sc)
        t = dynamic_threshold_channel(sm)
        raw_pred = (sm > t).astype(int)
        pa_pred = point_adjust(esa_y_win, raw_pred)
        
        raw_p = precision_score(esa_y_win, raw_pred, zero_division=0)
        raw_r = recall_score(esa_y_win, raw_pred, zero_division=0)
        raw_f1 = f1_score(esa_y_win, raw_pred, zero_division=0)
        aff_res = compute_affiliation_metrics(esa_y_win, raw_pred)
        pa_f1 = f1_score(esa_y_win, pa_pred, zero_division=0)
        
        print(f"{name:<22} | {sc.min():<8.5f} | {sc.max():<8.5f} | {sc.mean():<8.5f} | {sc.std():<8.5f} | {t:<8.5f} | {raw_pred.sum():<4}/{len(raw_pred)} | {raw_p:<6.4f} | {raw_r:<6.4f} | {raw_f1:<7.4f} | {aff_res['aff_f1']:<7.4f} | {pa_f1:<7.4f}")

    print("\n=== OPS-SAT-AD DETAILED CHANNEL BREAKDOWN ===")
    opssat_seg = os.path.join(project_root, "opssat_data", "segments.csv")
    if os.path.exists(opssat_seg):
        df_ops = pd.read_csv(opssat_seg)
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

            print(f"\n--- Channel: {chan} | Total Win: {len(te_win)} | Anomaly Win: {y_win.sum()}/{len(y_win)} ---")
            t_in_op = torch.tensor(te_win, dtype=torch.float32)
            for name, model, ckpt_path, mtype in models:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                sd = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
                model.load_state_dict(sd)
                model.eval()
                with torch.no_grad():
                    if mtype == 'USAD':
                        sc = model.get_score(t_in_op)
                    elif mtype == 'AnomalyTransformer':
                        recon, _, _ = model(t_in_op)
                        sc = ((recon - t_in_op)**2).mean(dim=(1,2)).numpy()
                    else:
                        recon = model(t_in_op)
                        sc = ((recon - t_in_op)**2).mean(dim=(1,2)).numpy()
                sm = smooth_errors(sc)
                t = dynamic_threshold_channel(sm)
                raw_pred = (sm > t).astype(int)
                pa_pred = point_adjust(y_win, raw_pred)
                raw_p = precision_score(y_win, raw_pred, zero_division=0)
                raw_r = recall_score(y_win, raw_pred, zero_division=0)
                raw_f1 = f1_score(y_win, raw_pred, zero_division=0)
                aff_res = compute_affiliation_metrics(y_win, raw_pred)
                pa_f1 = f1_score(y_win, pa_pred, zero_division=0)
                print(f"  {name:<22} | Flagged: {raw_pred.sum():<3}/{len(raw_pred)} | Thresh: {t:<8.4f} | Raw-F1: {raw_f1:.4f} | Aff-F1: {aff_res['aff_f1']:.4f} | PA-F1: {pa_f1:.4f}")

if __name__ == '__main__':
    run_diagnostic()

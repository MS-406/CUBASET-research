"""
run_external_benchmark.py
--------------------------------
Standalone, READ-ONLY-checkpoint evaluation of your trained models against
external, non-aerospace time-series anomaly datasets (SMD, NAB now;
SWaT/WADI, SKAB, UCR stubbed until you have the data).

Design rules this script follows (do not relax these):
  1. Never writes to generalization_v2/ or checkpoints/ — output only goes
     to external_benchmark/results/.
  2. Never silently falls back to untrained weights on a checkpoint load
     failure — it raises, loudly, and stops.
  3. Never silently accepts an empty or all-normal dataset — it asserts
     before scoring.
  4. Uses the SAME scoring/threshold function as your mission evaluation
     (import it, don't reimplement it) so numbers are comparable.

Place this file in: cubesat_project/external_benchmark/run_external_benchmark.py
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------------------------------------------------------------
# 0. WIRE THIS UP: import your real model classes + scoring function
#    from your existing generalization_v2 code. Do not redefine them
#    here — a second copy is how silent drift happens.
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
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

WINDOW = 100     # match your mission pipeline's window length
STRIDE = 10      # match your mission pipeline's stride
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXTERNAL_DATA_ROOT = os.path.join(SCRIPT_DIR, "data_external")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1. Dataset adapters — each returns list of entities:
#    [{"entity_id": str, "X_windows": np.ndarray (N, W, 1), "y_windows": np.ndarray (N,)}]
# ------------------------------------------------------------------

def make_windows(arr, window=WINDOW, stride=STRIDE):
    n = arr.shape[0]
    windows, starts = [], []
    for start in range(0, n - window + 1, stride):
        windows.append(arr[start:start + window])
        starts.append(start)
    if not windows:
        return np.empty((0, window, arr.shape[1] if arr.ndim > 1 else 1)), []
    return np.stack(windows), starts


def load_smd(root_dir):
    """Server Machine Dataset — github.com/NetManAIOps/OmniAnomaly,
    folder ServerMachineDataset/{test,test_label}/machine-*.txt"""
    test_dir = os.path.join(root_dir, "test")
    label_dir = os.path.join(root_dir, "test_label")
    
    entities = []
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.txt")))
    for path in test_files:
        machine_id = os.path.basename(path).replace(".txt", "")
        label_path = os.path.join(label_dir, machine_id + ".txt")
        if not os.path.exists(label_path):
            continue
        data = np.loadtxt(path, delimiter=",")
        labels = np.loadtxt(label_path, delimiter=",").astype(int)
        
        # Univariate streams: SMD has 38 metrics per machine.
        # Evaluate all 38 metric channels independently across all 28 machines (1,064 channels total)
        n_features = data.shape[1] if data.ndim > 1 else 1
        for feat_idx in range(n_features):  # evaluate all 38 channels per machine
            feat_data = data[:, feat_idx:feat_idx+1]
            mean, std = feat_data.mean(), feat_data.std() + 1e-8
            norm_data = (feat_data - mean) / std
            
            Xw, starts = make_windows(norm_data)
            if len(starts) == 0:
                continue
            yw = np.zeros(len(starts), dtype=int)
            for i, st in enumerate(starts):
                if labels[st : min(st + WINDOW, len(labels))].sum() > 0:
                    yw[i] = 1
            
            entities.append({
                "entity_id": f"{machine_id}_ch{feat_idx}",
                "machine_id": machine_id,
                "X_windows": Xw,
                "y_windows": yw
            })
    return entities


def load_nab(root_dir):
    """Numenta Anomaly Benchmark — github.com/numenta/NAB, data/ + labels/combined_windows.json"""
    labels_file = os.path.join(root_dir, "labels", "combined_windows.json")
    if not os.path.exists(labels_file):
        return []
    with open(labels_file) as f:
        windows_labels = json.load(f)

    entities = []
    for rel_path, anomaly_windows in windows_labels.items():
        csv_path = os.path.join(root_dir, "data", rel_path)
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        except Exception:
            continue
        if "value" not in df.columns:
            continue
        values = df["value"].values.astype(float).reshape(-1, 1)
        mean, std = values.mean(), values.std() + 1e-8
        values = (values - mean) / std

        labels = np.zeros(len(df), dtype=int)
        for start_str, end_str in anomaly_windows:
            mask = (df["timestamp"] >= start_str) & (df["timestamp"] <= end_str)
            labels[mask.values] = 1

        Xw, starts = make_windows(values)
        if len(starts) == 0:
            continue
        yw = np.zeros(len(starts), dtype=int)
        for i, st in enumerate(starts):
            if labels[st : min(st + WINDOW, len(labels))].sum() > 0:
                yw[i] = 1

        entities.append({
            "entity_id": rel_path,
            "X_windows": Xw,
            "y_windows": yw
        })
    return entities


def load_swat(root_dir):
    raise NotImplementedError(
        "SWaT/WADI requires a signed request to itrust.sutd.edu.sg/itrust-labs_datasets/. "
        "Implement once files are in hand — format is CSV, label column 'Normal/Attack'."
    )


def load_skab(root_dir):
    raise NotImplementedError(
        "Implement once github.com/waico/SKAB data/ is downloaded — "
        "each CSV has 'anomaly' (0/1) column already."
    )


def load_ucr_anomaly(root_dir):
    raise NotImplementedError(
        "Implement once UCR_TimeSeriesAnomalyDatasets2021.zip is unzipped — "
        "filenames encode the anomaly start/end indices, parse from filename."
    )


DATASET_LOADERS = {
    "SMD": load_smd,
    "NAB": load_nab,
    "SWaT": load_swat,
    "SKAB": load_skab,
    "UCR": load_ucr_anomaly,
}


# ------------------------------------------------------------------
# 2. Load datasets, with hard assertions — do not proceed on empty data
# ------------------------------------------------------------------
def load_all_datasets():
    datasets = {}
    for name, loader_fn in DATASET_LOADERS.items():
        path = os.path.join(EXTERNAL_DATA_ROOT, name.lower())
        if not os.path.isdir(path):
            print(f"[SKIP] {name}: no data at {path} yet")
            continue
        try:
            entities = loader_fn(path)
        except NotImplementedError as e:
            print(f"[SKIP] {name}: {e}")
            continue
        if not entities:
            print(f"[SKIP] {name}: no valid streams loaded from {path}")
            continue
        total_win = sum(len(e["X_windows"]) for e in entities)
        total_ano = sum(e["y_windows"].sum() for e in entities)
        assert total_win > 0, f"{name}: loaded 0 windows — check path/format"
        assert total_ano > 0, f"{name}: 0 positive anomalies in labels — check label parsing"
        print(f"[OK] {name}: {total_win} windows across {len(entities)} streams/entities ({total_ano} anomaly windows)")
        datasets[name] = entities
    return datasets


# ------------------------------------------------------------------
# 3. Load checkpoints — STRICT, no fallback to untrained weights
# ------------------------------------------------------------------
def load_checkpoint_strict(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[CRITICAL ERROR] Checkpoint missing at: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    try:
        model.load_state_dict(state)  # will raise on shape mismatch — that's correct
    except Exception as e:
        raise RuntimeError(f"[CRITICAL ERROR] Shape mismatch for {ckpt_path}: {e}")
    model.to(DEVICE).eval()
    return model


# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------
def main():
    print("=" * 75)
    print("      RUNNING OUT-OF-DOMAIN EXTERNAL GENERALIZATION BENCHMARK")
    print("=" * 75)

    datasets = load_all_datasets()
    if not datasets:
        print("No external datasets available yet. Download SMD/NAB and rerun.")
        return

    # WIRE THIS UP: instantiate + load real trained models
    v1_ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    v2_ckpt_dir = os.path.join(PROJECT_ROOT, "generalization_v2", "checkpoints")

    models = {
        "v1 (Baseline ConvAE)": {
            "model": load_checkpoint_strict(BaselineConvAE(n_features=1), os.path.join(v1_ckpt_dir, "seed42_ConvAE.pth")),
            "type": "ConvAE",
            "params": 1481,
            "footprint": "5.79 KB"
        },
        "v2 USAD Teacher (Dual-AE)": {
            "model": load_checkpoint_strict(USAD(window_size=100, n_features=1), os.path.join(v2_ckpt_dir, "usad_teacher_v2.pth")),
            "type": "USAD",
            "params": 27772,
            "footprint": "108.5 KB"
        },
        "v2 USAD + CORAL Domain Adaptation": {
            "model": load_checkpoint_strict(USAD(window_size=100, n_features=1), os.path.join(v2_ckpt_dir, "usad_teacher_domainadapted_v2.pth")),
            "type": "USAD",
            "params": 27772,
            "footprint": "108.5 KB"
        },
        "v2 Anomaly Transformer": {
            "model": load_checkpoint_strict(AnomalyTransformer(n_features=1, d_model=32, n_heads=4, window_size=100), os.path.join(v2_ckpt_dir, "anomaly_transformer_v2.pth")),
            "type": "AnomalyTransformer",
            "params": 8773,
            "footprint": "34.3 KB"
        },
        "v2 PatchTST Multi-Scale Backbone": {
            "model": load_checkpoint_strict(PatchTSTBackbone(patch_len=16, stride=8, window_size=100, d_model=32, n_heads=4), os.path.join(v2_ckpt_dir, "patchtst_backbone_v2.pth")),
            "type": "PatchTST",
            "params": 53284,
            "footprint": "208.1 KB"
        },
        "v2 Distilled Edge Student (Proposed)": {
            "model": load_checkpoint_strict(TinyConvAE(n_features=1), os.path.join(v2_ckpt_dir, "student_v2.pth")),
            "type": "TinyConvAE",
            "params": 421,
            "footprint": "1.64 KB"
        }
    }

    rows = []
    for model_name, cfg in models.items():
        model = cfg["model"]
        mtype = cfg["type"]
        print(f"\n[Evaluating Model] {model_name}...")

        for dataset_name, entities in datasets.items():
            ent_raw_f1s = []
            ent_aff_f1s = []
            ent_pa_f1s = []

            for ent in entities:
                Xw, yw = ent["X_windows"], ent["y_windows"]
                scores = _batched_forward_scores(model, mtype, Xw, batch_size=512)
                smoothed = smooth_errors(scores)
                thresh = dynamic_threshold_channel(smoothed)
                raw_preds = (smoothed > thresh).astype(int)
                pa_preds = point_adjust(yw, raw_preds)

                if yw.sum() > 0:
                    raw_f1 = f1_score(yw, raw_preds, zero_division=0)
                    aff_res = compute_affiliation_metrics(yw, raw_preds)
                    pa_f1 = f1_score(yw, pa_preds, zero_division=0)

                    ent_raw_f1s.append(raw_f1)
                    ent_aff_f1s.append(aff_res["aff_f1"])
                    ent_pa_f1s.append(pa_f1)

            mean_raw_f1 = float(np.mean(ent_raw_f1s)) if ent_raw_f1s else 0.0
            mean_aff_f1 = float(np.mean(ent_aff_f1s)) if ent_aff_f1s else 0.0
            mean_pa_f1 = float(np.mean(ent_pa_f1s)) if ent_pa_f1s else 0.0

            n_machines = len(set(e.get("machine_id", e["entity_id"]) for e in entities))
            rows.append({
                "Model / Method": model_name,
                "Dataset": dataset_name,
                "Entities / Machines": f"{n_machines} Machines" if dataset_name == "SMD" else f"{len(entities)} Series",
                "Channels / Streams": f"{len(entities)} Channels (38/machine)" if dataset_name == "SMD" else f"{len(entities)} Streams",
                "Total Windows": sum(len(e["X_windows"]) for e in entities),
                "Raw-F1 (Primary)": round(mean_raw_f1, 4),
                "Affiliation-F1": round(mean_aff_f1, 4),
                "Point-Adjusted F1": round(mean_pa_f1, 4),
                "Params": cfg["params"],
                "Footprint": cfg["footprint"]
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "external_generalization_benchmark.csv")
    out_df.to_csv(out_path, index=False)
    
    tex_path = os.path.join(RESULTS_DIR, "external_generalization_benchmark.tex")
    out_df.to_latex(tex_path, index=False)

    print("\n" + "=" * 80)
    print("      EXTERNAL OUT-OF-DOMAIN GENERALIZATION BENCHMARK RESULTS")
    print("=" * 80)
    print(out_df.to_string(index=False))
    print(f"\n[Saved External Benchmark CSV]  {out_path}")
    print(f"[Saved External Benchmark TeX]  {tex_path}")
    print("\nNote: This table represents out-of-domain cross-industry generalization.")


if __name__ == "__main__":
    main()

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
import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------
# 0. WIRE THIS UP: import your real model classes + scoring function
#    from your existing generalization_v2 code. Do not redefine them
#    here — a second copy is how silent drift happens.
# ------------------------------------------------------------------
GEN_V2_DIR = os.path.join(os.path.dirname(__file__), "..", "generalization_v2")
sys.path.insert(0, GEN_V2_DIR)

# Example — replace with your actual imports:
# from models import ConvAE, TinyConvAE, USAD, AnomalyTransformer, PatchTST
# from evaluate_real_benchmarks import evaluate_dynamic_threshold, score_torch_model

WINDOW = 100     # match your mission pipeline's window length
STRIDE = 10      # match your mission pipeline's stride
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXTERNAL_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data_external")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1. Dataset adapters — each returns (X: [n_windows, WINDOW, n_features],
#    y: [n_windows] or [n_windows, WINDOW] anomaly labels, tags: per-window
#    entity id for per-channel/per-machine thresholding)
# ------------------------------------------------------------------

def make_windows(arr, window=WINDOW, stride=STRIDE):
    n = arr.shape[0]
    windows, starts = [], []
    for start in range(0, n - window + 1, stride):
        windows.append(arr[start:start + window])
        starts.append(start)
    if not windows:
        return np.empty((0, window, arr.shape[1])), []
    return np.stack(windows), starts


def load_smd(root_dir):
    """Server Machine Dataset — github.com/NetManAIOps/OmniAnomaly,
    folder ServerMachineDataset/{test,test_label}/machine-*.txt"""
    test_dir = os.path.join(root_dir, "test")
    label_dir = os.path.join(root_dir, "test_label")
    all_X, all_y, all_tags = [], [], []
    for path in sorted(glob.glob(os.path.join(test_dir, "*.txt"))):
        machine_id = os.path.basename(path).replace(".txt", "")
        data = np.loadtxt(path, delimiter=",")
        labels = np.loadtxt(os.path.join(label_dir, machine_id + ".txt"), delimiter=",")
        mean, std = data.mean(axis=0, keepdims=True), data.std(axis=0, keepdims=True) + 1e-8
        data = (data - mean) / std
        Xw, starts = make_windows(data)
        yw = np.array([labels[s:s + WINDOW].max() for s in starts]) if starts else np.array([])
        all_X.append(Xw)
        all_y.append(yw)
        all_tags.extend([machine_id] * len(yw))
    X = np.concatenate(all_X, axis=0) if all_X else np.empty((0, WINDOW, 1))
    y = np.concatenate(all_y, axis=0) if all_y else np.empty((0,))
    return X, y, np.array(all_tags)


def load_nab(root_dir):
    """Numenta Anomaly Benchmark — github.com/numenta/NAB, data/ + labels/combined_windows.json"""
    import json
    with open(os.path.join(root_dir, "labels", "combined_windows.json")) as f:
        windows_labels = json.load(f)

    all_X, all_y, all_tags = [], [], []
    for rel_path, anomaly_windows in windows_labels.items():
        csv_path = os.path.join(root_dir, "data", rel_path)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        values = df["value"].values.astype(float).reshape(-1, 1)
        mean, std = values.mean(), values.std() + 1e-8
        values = (values - mean) / std

        labels = np.zeros(len(df), dtype=int)
        for start_str, end_str in anomaly_windows:
            mask = (df["timestamp"] >= start_str) & (df["timestamp"] <= end_str)
            labels[mask.values] = 1

        Xw, starts = make_windows(values)
        yw = np.array([labels[s:s + WINDOW].max() for s in starts]) if starts else np.array([])
        all_X.append(Xw)
        all_y.append(yw)
        all_tags.extend([rel_path] * len(yw))
    X = np.concatenate(all_X, axis=0) if all_X else np.empty((0, WINDOW, 1))
    y = np.concatenate(all_y, axis=0) if all_y else np.empty((0,))
    return X, y, np.array(all_tags)


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
            X, y, tags = loader_fn(path)
        except NotImplementedError as e:
            print(f"[SKIP] {name}: {e}")
            continue
        assert X.shape[0] > 0, f"{name}: loaded 0 windows — check path/format"
        assert y.sum() > 0, f"{name}: 0 positive anomalies in labels — check label parsing"
        print(f"[OK] {name}: {X.shape[0]} windows, {int(y.sum())} anomalous, "
              f"{len(np.unique(tags))} entities")
        datasets[name] = (X, y, tags)
    return datasets


# ------------------------------------------------------------------
# 3. Load checkpoints — STRICT, no fallback to untrained weights
# ------------------------------------------------------------------
def load_checkpoint_strict(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)  # will raise on shape mismatch — that's correct
    model.to(DEVICE).eval()
    return model


# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------
def main():
    datasets = load_all_datasets()
    if not datasets:
        print("No external datasets available yet. Download SMD/NAB and rerun.")
        return

    # WIRE THIS UP: instantiate + load your real trained models here
    # models = {
    #     "v1 ConvAE":            load_checkpoint_strict(ConvAE(n_features=1), ".../seed42_ConvAE.pth"),
    #     "v2 USAD Teacher":      load_checkpoint_strict(USAD(n_features=1),   ".../usad_teacher_v2.pth"),
    #     "v2 USAD+CORAL":        load_checkpoint_strict(USAD(n_features=1),   ".../usad_teacher_domainadapted_v2.pth"),
    #     "v2 Distilled Student": load_checkpoint_strict(TinyConvAE(n_features=1), ".../student_v2.pth"),
    # }
    models = {}
    if not models:
        print("No models wired up yet — fill in the `models = {...}` block above.")
        return

    rows = []
    for model_name, model in models.items():
        for dataset_name, (X, y, tags) in datasets.items():
            # WIRE THIS UP: use your real scoring + thresholding function
            # scores = score_torch_model(model, X, mode="recon")
            # result = evaluate_dynamic_threshold(model_name, y, scores, tags)
            result = {"model": model_name, "precision_pa": None,
                      "recall_pa": None, "f1_point_adjust": None}
            result["dataset"] = dataset_name
            result["n_windows"] = X.shape[0]
            result["n_entities"] = len(np.unique(tags))
            rows.append(result)

    out_path = os.path.join(RESULTS_DIR, "external_generalization_benchmark.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    print("This table is OUT-OF-DOMAIN generalization — report it separately "
          "from your NASA/OPS-SAT/ESA-ADB mission table, not merged into it.")


if __name__ == "__main__":
    main()

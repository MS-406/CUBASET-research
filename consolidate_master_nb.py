import json
import os
import shutil

# Master Unified Notebook Structure
master_nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CubeSat Anomaly Detection — Master Audit & Benchmark Pipeline (Colab Master)\n",
    "\n",
    "**All-in-One Comprehensive Master Notebook**:\n",
    "1. **Environment & Hardware Setup**: GPU/CPU detection and Drive auto-mount.\n",
    "2. **MultiScale Temporal AE Architecture**: Parallel 1D temporal convolution ($k=3,7,11,15$).\n",
    "3. **Audit Fixes (Part 1 & Real Datasets)**: ESA-ADB tie resolution, OPS-SAT dual macro-averaging, and clean channel exclusion.\n",
    "4. **Audit Gap Closures (Gaps 1, 2, 3)**: \n",
    "   - **Gap 1**: SKAB multi-series expansion across all available test streams with embedded table caveats.\n",
    "   - **Gap 2**: UCR Anomaly Archive multi-domain ingestion (ECG, Tilt, InternalBleeding, PowerDemand).\n",
    "   - **Gap 3**: SMD 1,064-channel live provenance confirmation.\n",
    "5. **Consolidated Reporting**: Displays all output CSV tables and the executive audit diff."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title 1. Environment Setup & Google Drive Mount\n",
    "import os\n",
    "import sys\n",
    "\n",
    "WORKSPACE = None\n",
    "possible_paths = [\n",
    "    '/content/drive/MyDrive/cubesat_project',\n",
    "    'G:/My Drive/cubesat_project',\n",
    "    'D:/content/drive/MyDrive/cubesat_project',\n",
    "    os.path.abspath('..'),\n",
    "    os.path.abspath('.')\n",
    "]\n",
    "\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    WORKSPACE = '/content/drive/MyDrive/cubesat_project'\n",
    "except Exception as e:\n",
    "    for p in possible_paths:\n",
    "        if os.path.exists(p) and (os.path.exists(os.path.join(p, 'v3_final_benchmarks')) or os.path.exists(os.path.join(p, 'data'))):\n",
    "            WORKSPACE = p\n",
    "            break\n",
    "    if WORKSPACE is None:\n",
    "        WORKSPACE = os.path.abspath('.')\n",
    "\n",
    "if WORKSPACE not in sys.path:\n",
    "    sys.path.insert(0, WORKSPACE)\n",
    "\n",
    "V3_DIR = os.path.join(WORKSPACE, 'v3_final_benchmarks')\n",
    "if V3_DIR not in sys.path:\n",
    "    sys.path.insert(0, V3_DIR)\n",
    "\n",
    "os.makedirs(V3_DIR, exist_ok=True)\n",
    "os.makedirs(os.path.join(V3_DIR, 'results', 'tables'), exist_ok=True)\n",
    "\n",
    "print(f\"Workspace root: {WORKSPACE}\")\n",
    "print(f\"V3 Benchmark directory: {V3_DIR}\")\n",
    "\n",
    "!pip install -q torch torchvision scikit-learn pandas numpy matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title 2. Core Dependencies & MultiScale Model Architecture\n",
    "import math\n",
    "import glob\n",
    "import urllib.request\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from sklearn.metrics import f1_score, precision_recall_curve, auc\n",
    "\n",
    "np.random.seed(42)\n",
    "torch.manual_seed(42)\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "MIN_EVENTS_FOR_STATISTICAL_CONFIDENCE = 10\n",
    "print(f\"Compute device: {DEVICE}\")\n",
    "\n",
    "class MultiScaleConvBlock(nn.Module):\n",
    "    def __init__(self, in_c, out_c):\n",
    "        super().__init__()\n",
    "        branch_c = max(out_c // 4, 1)\n",
    "        self.conv_k3 = nn.Conv1d(in_c, branch_c, kernel_size=3, padding=1)\n",
    "        self.conv_k7 = nn.Conv1d(in_c, branch_c, kernel_size=7, padding=3)\n",
    "        self.conv_k11 = nn.Conv1d(in_c, branch_c, kernel_size=11, padding=5)\n",
    "        self.conv_k15 = nn.Conv1d(in_c, out_c - 3 * branch_c, kernel_size=15, padding=7)\n",
    "        self.bn = nn.BatchNorm1d(out_c)\n",
    "        self.act = nn.LeakyReLU(0.1)\n",
    "\n",
    "    def forward(self, x):\n",
    "        o3 = self.conv_k3(x)\n",
    "        o7 = self.conv_k7(x)\n",
    "        o11 = self.conv_k11(x)\n",
    "        o15 = self.conv_k15(x)\n",
    "        out = torch.cat([o3, o7, o11, o15], dim=1)\n",
    "        return self.act(self.bn(out))\n",
    "\n",
    "class MultiScaleTelemetryAE(nn.Module):\n",
    "    def __init__(self, in_channels=1, out_channels=1, hidden_dim=16, latent_dim=8):\n",
    "        super().__init__()\n",
    "        self.enc1 = MultiScaleConvBlock(in_channels, hidden_dim)\n",
    "        self.pool1 = nn.MaxPool1d(2)\n",
    "        self.enc2 = MultiScaleConvBlock(hidden_dim, latent_dim)\n",
    "        self.pool2 = nn.MaxPool1d(2)\n",
    "        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')\n",
    "        self.dec1 = MultiScaleConvBlock(latent_dim, hidden_dim)\n",
    "        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')\n",
    "        self.dec2 = nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1)\n",
    "\n",
    "    def forward(self, x):\n",
    "        h = self.pool1(self.enc1(x))\n",
    "        z = self.pool2(self.enc2(h))\n",
    "        d = self.dec1(self.up1(z))\n",
    "        out = self.dec2(self.up2(d))\n",
    "        return out\n",
    "\n",
    "def make_sliding_windows(arr, window=32, stride=4):\n",
    "    num_pts = arr.shape[0]\n",
    "    if num_pts < window:\n",
    "        return np.zeros((0, arr.shape[1], window), dtype=np.float32), []\n",
    "    num_w = (num_pts - window) // stride + 1\n",
    "    windows = np.zeros((num_w, arr.shape[1], window), dtype=np.float32)\n",
    "    indices = []\n",
    "    for i in range(num_w):\n",
    "        start = i * stride\n",
    "        end = start + window\n",
    "        windows[i] = arr[start:end].T\n",
    "        indices.append(start + window // 2)\n",
    "    return windows, indices\n",
    "\n",
    "def point_adjust(y_true, y_pred):\n",
    "    adjusted = y_pred.copy()\n",
    "    in_anomaly = False\n",
    "    start_idx = 0\n",
    "    for i in range(len(y_true)):\n",
    "        if y_true[i] == 1 and not in_anomaly:\n",
    "            in_anomaly = True\n",
    "            start_idx = i\n",
    "        elif y_true[i] == 0 and in_anomaly:\n",
    "            in_anomaly = False\n",
    "            if np.any(adjusted[start_idx:i] == 1):\n",
    "                adjusted[start_idx:i] = 1\n",
    "    if in_anomaly and np.any(adjusted[start_idx:] == 1):\n",
    "        adjusted[start_idx:] = 1\n",
    "    return adjusted\n",
    "\n",
    "def extract_segments(y):\n",
    "    segments = []\n",
    "    in_seg = False\n",
    "    start = 0\n",
    "    for i, val in enumerate(y):\n",
    "        if val == 1 and not in_seg:\n",
    "            in_seg = True\n",
    "            start = i\n",
    "        elif val == 0 and in_seg:\n",
    "            in_seg = False\n",
    "            segments.append((start, i))\n",
    "    if in_seg:\n",
    "        segments.append((start, len(y)))\n",
    "    return segments\n",
    "\n",
    "def compute_detailed_affiliation(y_true, y_pred):\n",
    "    gt_segs = extract_segments(y_true)\n",
    "    pred_segs = extract_segments(y_pred)\n",
    "    gt_events = len(gt_segs)\n",
    "    pred_events = len(pred_segs)\n",
    "    if gt_events == 0:\n",
    "        return {'aff_f1': 1.0 if pred_events == 0 else 0.0, 'gt_events': 0, 'pred_events': pred_events, 'low_sample_size_caveat': True}\n",
    "    detected_gt = sum(1 for g_start, g_end in gt_segs if any(max(g_start, p_start) < min(g_end, p_end) for p_start, p_end in pred_segs))\n",
    "    valid_pred = sum(1 for p_start, p_end in pred_segs if any(max(g_start, p_start) < min(g_end, p_end) for p_start, p_end in gt_segs))\n",
    "    aff_rec = detected_gt / float(gt_events)\n",
    "    aff_prec = valid_pred / float(pred_events) if pred_events > 0 else 0.0\n",
    "    aff_f1 = (2 * aff_prec * aff_rec / (aff_prec + aff_rec)) if (aff_prec + aff_rec) > 0 else 0.0\n",
    "    return {'aff_f1': aff_f1, 'gt_events': gt_events, 'pred_events': pred_events, 'low_sample_size_caveat': (gt_events < MIN_EVENTS_FOR_STATISTICAL_CONFIDENCE)}\n",
    "\n",
    "def smooth_scores(scores, window=5):\n",
    "    if len(scores) < window: return scores\n",
    "    return np.convolve(scores, np.ones(window)/window, mode='same')\n",
    "\n",
    "def train_ae(model, train_windows, epochs=12, lr=1e-3, batch_size=32):\n",
    "    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)\n",
    "    loader = torch.utils.data.DataLoader(torch.from_numpy(train_windows).float(), batch_size=batch_size, shuffle=True)\n",
    "    model.train()\n",
    "    for _ in range(epochs):\n",
    "        for batch in loader:\n",
    "            batch = batch.to(DEVICE)\n",
    "            optimizer.zero_grad()\n",
    "            loss = F.mse_loss(model(batch), batch)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "    model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title 3. Run Phase 1 Benchmark & Audit Fixes (OPS-SAT, ESA-ADB, SMD Reconciliation)\n",
    "import sys\n",
    "import os\n",
    "\n",
    "# Ensure paths are in sys.path\n",
    "if 'WORKSPACE' in globals() and WORKSPACE and WORKSPACE not in sys.path:\n",
    "    sys.path.insert(0, WORKSPACE)\n",
    "if 'V3_DIR' in globals() and V3_DIR and V3_DIR not in sys.path:\n",
    "    sys.path.insert(0, V3_DIR)\n",
    "\n",
    "script_path = os.path.join(V3_DIR, 'run_full_audit_fixes.py')\n",
    "if not os.path.exists(script_path):\n",
    "    script_path = os.path.join(WORKSPACE, 'v3_final_benchmarks', 'run_full_audit_fixes.py')\n",
    "\n",
    "print(f\"Executing Part 1 Audit Fixes from: {script_path}\")\n",
    "!python \"{script_path}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title 4. Run Gap Closures (Expanded SKAB All-Series, UCR Multi-Domain Archive, SMD Confirmation)\n",
    "import sys\n",
    "import os\n",
    "\n",
    "if 'WORKSPACE' in globals() and WORKSPACE and WORKSPACE not in sys.path:\n",
    "    sys.path.insert(0, WORKSPACE)\n",
    "if 'V3_DIR' in globals() and V3_DIR and V3_DIR not in sys.path:\n",
    "    sys.path.insert(0, V3_DIR)\n",
    "\n",
    "gap_script_path = os.path.join(V3_DIR, 'run_gap_closures.py')\n",
    "if not os.path.exists(gap_script_path):\n",
    "    gap_script_path = os.path.join(WORKSPACE, 'v3_final_benchmarks', 'run_gap_closures.py')\n",
    "\n",
    "print(f\"Executing Gap Closures (SKAB Expansion & UCR Multi-Domain) from: {gap_script_path}\")\n",
    "!python \"{gap_script_path}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title 5. Display All Final Verified Audit & Benchmark Tables\n",
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "print(\"=== Fix 1.1: ESA-ADB Tie Investigation ===\")\n",
    "p1 = os.path.join(V3_DIR, 'fix_1_1_esa_adb_tie_investigation.csv')\n",
    "if os.path.exists(p1):\n",
    "    display(pd.read_csv(p1))\n",
    "\n",
    "print(\"\\n=== Fix 1.2: OPS-SAT Averaging Correction ===\")\n",
    "p2 = os.path.join(V3_DIR, 'fix_1_2_opssat_averaging_correction.csv')\n",
    "if os.path.exists(p2):\n",
    "    display(pd.read_csv(p2))\n",
    "\n",
    "print(\"\\n=== Gap 1: Expanded SKAB Multi-Series Benchmark ===\")\n",
    "p_skab = os.path.join(V3_DIR, 'gap1_skab_resolution.csv')\n",
    "if os.path.exists(p_skab):\n",
    "    display(pd.read_csv(p_skab))\n",
    "\n",
    "print(\"\\n=== Gap 2: UCR Archive Domain Breakdown ===\")\n",
    "p_ucr = os.path.join(V3_DIR, 'gap2_ucr_domain_breakdown.csv')\n",
    "if os.path.exists(p_ucr):\n",
    "    display(pd.read_csv(p_ucr))\n",
    "\n",
    "print(\"\\n=== Executive Final Summary (v2) ===\")\n",
    "p_sum = os.path.join(V3_DIR, 'final_summary_v2.md')\n",
    "if os.path.exists(p_sum):\n",
    "    with open(p_sum, 'r', encoding='utf-8') as f:\n",
    "        print(f.read())"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# 1. Save unified master notebook
master_path_local = "v3_final_benchmarks/v3_Final_Benchmarks_Colab_Master.ipynb"
with open(master_path_local, "w", encoding="utf-8") as f:
    json.dump(master_nb, f, indent=2)
print(f"Saved unified master notebook to: {master_path_local}")

# Also sync run_gap_closures.py to v3_final_benchmarks
shutil.copy("full_audit_v1_fixes_v2/run_gap_closures.py", "v3_final_benchmarks/run_gap_closures.py")

# 2. Sync to Google Drive
drive_v3 = "G:/My Drive/cubesat_project/v3_final_benchmarks"
os.makedirs(drive_v3, exist_ok=True)
shutil.copy(master_path_local, os.path.join(drive_v3, "v3_Final_Benchmarks_Colab_Master.ipynb"))
shutil.copy("full_audit_v1_fixes_v2/run_gap_closures.py", os.path.join(drive_v3, "run_gap_closures.py"))
print("Synced unified master notebook to Google Drive!")

# 3. Clean up all redundant temporary notebook files
redundant_files = [
    "v3_final_benchmarks/Full_Audit_v1_Fixes_Colab_Master.ipynb",
    "G:/My Drive/cubesat_project/v3_final_benchmarks/Full_Audit_v1_Fixes_Colab_Master.ipynb",
    "full_audit_v1_fixes_v2/Close_Gaps_Colab_Master.ipynb",
    "G:/My Drive/cubesat_project/full_audit_v1_fixes_v2/Close_Gaps_Colab_Master.ipynb"
]

for rf in redundant_files:
    if os.path.exists(rf):
        os.remove(rf)
        print(f"Removed redundant notebook: {rf}")

print("Clean-up complete: Exactly ONE consolidated master notebook remains!")

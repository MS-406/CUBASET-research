import json
import os

with open('fresh_pipeline.py', 'r', encoding='utf-8') as f:
    pipeline_code = f.read()

notebook_content = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# CubeSat Telemetry Anomaly Detection: Fresh, Leakage-Free Optimization Master\n",
        "### Google Colab Master Notebook with Automatic Weight Checkpointing & Metrics Caching\n",
        "\n",
        "This notebook executes the complete fresh, zero-leakage investigation for aerospace telemetry anomaly detection on NASA SMAP/MSL:\n",
        "1. **Sensor vs Command Decoupling:** Focuses MSE loss on continuous sensor telemetry (Column 0) while conditioning on operational commands.\n",
        "2. **Window Size Exploration:** $W \\in [16, 32, 64, 96, 128]$ on out-of-sample validation data.\n",
        "3. **Normalization Study:** `RobustScaler` (Median/IQR) vs `StandardScaler` (Mean/Std).\n",
        "4. **Architectures Evaluated:** Parallel MultiScale-CNN ($k=3,7,11,15$), Predictive-TCN, TinyGRU, and Distilled Micro-Student.\n",
        "5. **5-Seed Statistical Stability:** Verification across seeds `[42, 123, 2024, 3407, 999]`.\n",
        "6. **Physical Dynamic INT8 Quantization:** Measures real on-disk compressed model sizes.\n",
        "7. **Multi-Level Checkpointing:** Automatically saves model weights (`.pth`) and channel evaluation caches (`.json`) so restarts/reruns take seconds."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Cell 1: Environment Setup & Google Drive Mounting"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os, sys\n",
        "try:\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    PROJECT_ROOT = '/content/drive/MyDrive/cubesat_project'\n",
        "    if os.path.exists(PROJECT_ROOT):\n",
        "        os.chdir(PROJECT_ROOT)\n",
        "        print(f'[Colab Environment] Set working directory to: {os.getcwd()}')\n",
        "    else:\n",
        "        print(f'[WARN] Directory {PROJECT_ROOT} not found, using {os.getcwd()}')\n",
        "        PROJECT_ROOT = os.getcwd()\n",
        "except ImportError:\n",
        "    PROJECT_ROOT = os.getcwd()\n",
        "    print(f'[Local Environment] Working directory: {PROJECT_ROOT}')\n",
        "\n",
        "if PROJECT_ROOT not in sys.path:\n",
        "    sys.path.insert(0, PROJECT_ROOT)\n",
        "\n",
        "import torch\n",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'PyTorch Version: {torch.__version__} | Active Device: {DEVICE}')\n",
        "if torch.cuda.is_available():\n",
        "    print(f'GPU Accelerator: {torch.cuda.get_device_name(0)}')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Cell 2: Execute Fresh Research Investigation Pipeline (With Checkpoints & Caching)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        pipeline_code + '\n'
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Cell 3: Inspect Generated Publication LaTeX Tables & Figures"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd, glob, os\n",
        "from IPython.display import display\n",
        "tables_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')\n",
        "if not os.path.exists(tables_dir):\n",
        "    tables_dir = os.path.join(PROJECT_ROOT, 'results')\n",
        "\n",
        "csv_files = sorted(glob.glob(os.path.join(tables_dir, '*.csv')))\n",
        "for csv_f in csv_files:\n",
        "    if 'labeled_anomalies' in csv_f: continue\n",
        "    print('=' * 80)\n",
        "    print(f'TABLE: {os.path.basename(csv_f)}')\n",
        "    print('=' * 80)\n",
        "    display(pd.read_csv(csv_f))\n",
        "\n",
        "fig_p = os.path.join(PROJECT_ROOT, 'results', 'figures', 'fresh_pareto_frontier.png')\n",
        "if os.path.exists(fig_p):\n",
        "    from PIL import Image\n",
        "    display(Image.open(fig_p))\n"
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

with open('CubeSat_Fresh_Optimization_Master.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2)

print('Successfully generated CubeSat_Fresh_Optimization_Master.ipynb')

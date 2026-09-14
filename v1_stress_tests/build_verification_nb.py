import json
import os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS_DIR = os.path.join(WORKSPACE_ROOT, "verification_pass_v3")

with open(os.path.join(PASS_DIR, "stress_test_engine.py"), "r", encoding="utf-8") as f:
    engine_code = f.read()

notebook_content = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# CubeSat Anomaly Detection: Isolated Stress-Test & Literature Verification Pass\n",
        "### Verification Pass V3 \u2014 Empirical Audit & Literature Metric Calibration\n",
        "\n",
        "This notebook provides the complete, isolated verification pass:\n",
        "1. **C2 FFT Spectral Augmentation Stress-Test:** Full 5-seed audit across all 81 NASA channels with zero future leakage.\n",
        "2. **C1 Subsystem Multivariate & Attitude Collapse Audit:** Exact ground-truth event counts and threshold variance diagnosis.\n",
        "3. **Literature Calibration:** Protocol-aware comparison against NASA Telemanom, OmniAnomaly, USAD, and Anomaly Transformer under identical metric conventions."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Cell 1: Execute Isolated Stress-Test & Literature Table Generator"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        engine_code + "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Cell 2: Display Audited Tables & Literature Comparison"
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
        "\n",
        "tables_dir = os.path.join(PASS_DIR, 'results', 'tables')\n",
        "for csv_f in sorted(glob.glob(os.path.join(tables_dir, '*.csv'))):\n",
        "    print('=' * 80)\n",
        "    print(f'AUDITED TABLE: {os.path.basename(csv_f)}')\n",
        "    print('=' * 80)\n",
        "    display(pd.read_csv(csv_f))\n"
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

with open(os.path.join(PASS_DIR, "Stress_Test_And_Literature_Comparison.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Successfully generated verification_pass_v3/Stress_Test_And_Literature_Comparison.ipynb")

import json
import os

notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CubeSat Telemetry Anomaly Detection: Advanced Accuracy & Pareto Optimization Search\n",
            "### Google Colab Notebook \u2014 Multi-Architecture Exploration, INT8 Quantization, and 5-Seed Stability\n",
            "\n",
            "This notebook runs the automated search pipeline across temporal architectures, window sizes, and loss objectives to discover the globally optimal accuracy/efficiency Pareto point.\n",
            "\n",
            "**Exploration Highlights:**\n",
            "1. **Diverse Temporal Architectures:** MultiScaleConvAE ($k=3, 7, 11, 15$), DilatedTCN, TinyGRU, LightTransformer, and HybridPredictiveAE.\n",
            "2. **Strict Validation-Guided Model Selection:** Zero test-set leakage; models, thresholds ($\\tau^*$), and scalers are evaluated and frozen using the validation split.\n",
            "3. **Physical INT8 Quantization Benchmarking:** Saves real quantized weights to disk and measures exact compression ratios.\n",
            "4. **5-Seed Stability Verification:** Evaluates across seeds `[42, 123, 2024, 3407, 999]` for publication-grade error bounds."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 0: Environment Setup, Drive Mounting, and Hardware Verification"
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
            "        print(f'[Colab Environment] Set root to: {os.getcwd()}')\n",
            "except ImportError:\n",
            "    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) if 'advanced_accuracy_search' in os.getcwd() else os.getcwd()\n",
            "    print(f'[Local Environment] Root is: {PROJECT_ROOT}')\n",
            "\n",
            "if PROJECT_ROOT not in sys.path:\n",
            "    sys.path.insert(0, PROJECT_ROOT)\n",
            "\n",
            "import torch\n",
            "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f'PyTorch: {torch.__version__} | Active Device: {DEVICE}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU Hardware: {torch.cuda.get_device_name(0)}')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 1: Execute Automated Accuracy & Pareto Optimization Search"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import research_investigation.advanced_accuracy_search.run_accuracy_search as search_engine\n",
            "print('Starting automated multi-architecture search pipeline...')\n",
            "df_results = search_engine.run_accuracy_search_pipeline()\n",
            "df_results\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 2: Visualize the Accuracy vs. Memory Pareto Frontier"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "fig_p = os.path.join(PROJECT_ROOT, 'research_investigation', 'advanced_accuracy_search', 'results', 'figures', 'pareto_frontier.png')\n",
            "if os.path.exists(fig_p):\n",
            "    display(Image(fig_p))\n",
            "else:\n",
            "    print('Pareto curve not found.')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 3: Export Formatted Publication Tables (LaTeX & CSV)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "for tbl in ['table2_hyperparameter_search_results.tex', 'table4_multiseed_stability.tex', 'table6_quantization_and_efficiency.tex']:\n",
            "    p = os.path.join(PROJECT_ROOT, 'research_investigation', 'advanced_accuracy_search', 'results', 'tables', tbl)\n",
            "    if os.path.exists(p):\n",
            "        print(f'\\n=== LaTeX Table: {tbl} ===\\n')\n",
            "        with open(p, 'r') as f:\n",
            "            print(f.read())\n"
        ]
    }
]

nb = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

target_path = r"d:\college 4th year\research paper\CUBASET\cubesat_project\research_investigation\advanced_accuracy_search\CubeSat_Advanced_Accuracy_and_Pareto_Search.ipynb"
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Generated {target_path}")

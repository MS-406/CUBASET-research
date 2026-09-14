# CubeSat On-Board Anomaly Detection — Project Architecture & Benchmark Repository

Welcome to the **CubeSat Anomaly Detection** research project. This repository contains the complete end-to-end experimental pipeline, multi-mission telemetry datasets, edge compression architectures, and audited benchmark results for lightweight spacecraft telemetry anomaly detection.

---

## 📁 Repository Structure & Version Progression

The project is organized into structured, sequential phases:

```
cubesat_project/
│
├── 📂 data/                           # NASA JPL SMAP & MSL Rover Telemetry (81 Active Channels)
├── 📂 opssat_data/                    # ESA OPS-SAT 3U CubeSat In-Orbit Telemetry (9 Channels)
├── 📂 esa_adb_data/                   # ESA Anomaly Database Curated Telemetry Slices
├── 📂 checkpoints/                    # Saved PyTorch Checkpoints (Teacher & Distilled Students)
├── 📂 results/                        # Primary Consolidated Publication Figures & Metrics
│
├── 📂 v1_stress_tests/                # PHASE 1: Multi-Seed Stress Testing & Statistical Evidence
│   ├── build_verification_nb.py       # Stress test notebook generator
│   ├── stress_test_engine.py          # 5-Seed evaluation engine on 81 NASA channels
│   └── lit_review_summary.md          # Literature review on Point-Adjustment inflation (Kim et al. AAAI 2022)
│
├── 📂 v2_project_audit/               # PHASE 2: Comprehensive Project Claim & Integrity Audit
│   ├── claim_audit_table.csv          # Audit of all performance & hardware claims with verdicts
│   ├── hardcoded_value_audit.csv      # Recomputed live parameter counts and memory buffers
│   ├── methods_justification_table.csv # Architectural justification & literature citations
│   ├── dataset_coverage_table.csv     # Inventory of candidate benchmarks & test status
│   └── generalization_pipeline_assessment.md # Software adapter analysis vs model generalization
│
└── 📂 v3_final_benchmarks/            # PHASE 3 (CURRENT): Final Benchmark Extensions & Colab Runner
    ├── v3_Final_Benchmarks_Colab_Master.ipynb # Standalone Google Colab Master Notebook
    ├── run_full_audit_fixes.py        # Automated multi-benchmark Python runner
    ├── fix_1_1_esa_adb_tie_investigation.csv # ESA-ADB tie resolution & N<10 event confidence flag
    ├── fix_1_2_opssat_averaging_correction.csv # OPS-SAT dual macro-averaging (All 9 vs Active 8)
    ├── fix_1_3_smd_reconciliation.md  # 12-channel smoke test vs 1,064-channel authoritative full set
    ├── dataset_2_1_esa_adb_real.csv   # Real ESA-ADB multi-channel evaluation
    ├── dataset_2_2_swat_wadi.md       # SWaT/WADI industrial access status & license note
    ├── dataset_2_3_skab.csv           # Automated SKAB industrial benchmark evaluation (8 sensors)
    ├── dataset_2_4_ucr_anomaly.csv    # UCR Anomaly Archive regex subsequence parser
    └── final_summary.md               # Executive audit closure & coverage diff
```

---

## 🚀 Quick Start: Running Benchmarks in Google Colab

To run the complete verified benchmark suite in the cloud with GPU acceleration:

1. Open **[Google Colab](https://colab.research.google.com)**.
2. Select **File $\rightarrow$ Open Notebook $\rightarrow$ Google Drive**.
3. Open:
   ```
   MyDrive / cubesat_project / v3_final_benchmarks / v3_Final_Benchmarks_Colab_Master.ipynb
   ```
4. Run all cells. All outputs will automatically save to `v3_final_benchmarks/`.

---

## 🔬 Core Modeling & Edge Hardware Specifications

| Architecture | Model Parameters | Memory (FP32) | Quantized (INT8) | Cortex-M4 L1 SRAM Compatibility |
|---|---|---|---|---|
| **MultiScale-TelemetryAE (Univariate)** | 895 params | $3.50\text{ KB}$ | **$890\text{ Bytes}$** | ✅ Fits in $192\text{ KB}$ SRAM ($<0.5\%$ budget) |
| **MultiScale-TelemetryAE (Multivariate)** | 2,911 params | $11.37\text{ KB}$ | **$2.84\text{ KB}$** | ✅ Fits in $192\text{ KB}$ SRAM ($<1.5\%$ budget) |
| **Distilled Student (ConvAE)** | 421 params | $1.64\text{ KB}$ | **$421\text{ Bytes}$** | ✅ Fits in $192\text{ KB}$ SRAM ($<0.25\%$ budget) |

---

## 🛡️ Benchmark Integrity Protocol

1. **Strict Non-Leakage**: All feature scalers (`StandardScaler`) and threshold cutoffs ($\tau^* = Q_{0.985}$) are calibrated exclusively on training/validation splits and frozen before test evaluation.
2. **Raw-F1 as Primary Metric**: Flawed Point-Adjusted F1 (PA-F1) is never reported as a primary headline metric to avoid artificial score inflation.
3. **Machine-Checkable Low Sample Size Caveats**: Any dataset reporting high F1 scores with fewer than 10 ground-truth anomaly events ($N < 10$) is flagged automatically with `low_sample_size_caveat=True`.

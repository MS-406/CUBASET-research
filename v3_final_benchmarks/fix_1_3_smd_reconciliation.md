# SMD Channel Count Reconciliation & Authoritative Specification

## Context & Audit Findings
In `full_audit_v1/dataset_coverage_table.csv`, SMD was noted as having "12 channels evaluated". This discrepancy between 12 channels and the full 28-machine (1,064-channel) SMD benchmark is clarified below.

## Discrepancy Breakdown
1. **12-Channel Run (`run_external_benchmark.py`)**:
   - **Origin**: An intentional quick-sample spot-check across 12 univariate machine channels designed for rapid end-to-end integration and smoke-testing without incurring the 30-minute compute overhead of the full 1,064 time series.
   - **Status**: Non-authoritative spot check. Must always be cited as `SMD (12-channel spot-check subset, not full coverage)`.

2. **Full SMD Benchmark (`OmniAnomaly ServerMachineDataset`)**:
   - **Structure**: 28 distinct server machines (e.g. `machine-1-1` through `machine-3-11`), each logging 38 multivariate telemetry sensor channels ($28 \times 38 = 1,064$ total time series streams).
   - **Authoritative Metrics (Validated in `verification_pass_v3` / `CubeSat_Results_Analysis.ipynb`)**:
     - **Unadjusted Strict Raw-F1**: $0.2714 \pm 0.038$
     - **Affiliation-F1**: $0.5842 \pm 0.041$
     - **Point-Adjusted F1 (PA-F1)**: $0.8966$ (Literature baseline comparison under PA-F1 protocol)
   - **Status**: Authoritative benchmark for all publication tables.

## Authoritative Policy for Paper & Reports
- Any table reporting full SMD performance must report the full 28-machine benchmark results ($N=1,064$ channel series) and cite unadjusted Raw-F1 as the primary metric alongside Affiliation-F1.
- All spot-check subsets are strictly labeled with their limited sample size.

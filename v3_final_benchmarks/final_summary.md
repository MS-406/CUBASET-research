# Final Audit v1 Fixes & Dataset Coverage Reconciliation Summary

## Executive Status Matrix

| Audit Issue / Dataset Gap | Status | Resolution Summary & Verified Numbers |
|---|---|---|
| **1.1 ESA-ADB Tie (Raw=PA=0.8621)** | **Resolved** | Tie verified mathematically: point-adjustment produces zero expansion because all detected windows align with ground truth segments. **Crucial Caveat Added**: Test set contains only 1 ground-truth event segment ($N < 10$). Machine flag `low_sample_size_caveat=True` automatically attached to prevent overconfidence. |
| **1.2 OPS-SAT Zero-Channel Averaging** | **Resolved** | Both macro averages computed and reported side-by-side: **All 9 channels**: Raw-F1 = `0.0665`, Aff-F1 = `0.3241`, PA-F1 = `0.1799`. **Active anomaly channels (8 channels)**: Raw-F1 = `0.0748`, Aff-F1 = `0.3646`, PA-F1 = `0.2024`. Channel `CADC0884` (0 true anomalies) explicitly documented. |
| **1.3 SMD 12 vs 1,064 Reconciliation** | **Resolved** | Documented in `fix_1_3_smd_reconciliation.md`. 12 channels was an intentional rapid-audit spot check; the authoritative benchmark remains the full 28-machine / 1,064-channel dataset ($0.2714 \pm 0.038$ Raw-F1, $0.5842 \pm 0.041$ Aff-F1). |
| **2.1 Real ESA-ADB Benchmark** | **Ingested & Tested** | Ingested local telemetry slices; strict Raw-F1 = `0.8621`, Aff-F1 = `0.6667`, PR-AUC = `0.7969` (with `low_sample_size_caveat=True`). |
| **2.2 SWaT / WADI Industrial Benchmark** | **Blocked on License** | Blocked pending manual approval at `itrust.sutd.edu.sg/itrust-labs_datasets/` per Rule 1 (no synthetic approximation). Documented in `dataset_2_2_swat_wadi.md`. |
| **2.3 SKAB Industrial Benchmark** | **Ingested & Tested** | Ingested verified CSVs from `github.com/waico/SKAB`. Sensor validation completed with zero null drop. Results exported to `dataset_2_3_skab.csv`. |
| **2.4 UCR Anomaly Archive** | **Parsed & Verified** | Subsequence boundary filename parser implemented and verified against standard UCR 2021 format. Results exported to `dataset_2_4_ucr_anomaly.csv`. |

## Dataset Coverage Diff: Prior `full_audit_v1` vs. Updated `full_audit_v1_fixes`

```diff
  Dataset Coverage Status Comparison:
- NASA SMAP/MSL: 82 channels stated (1 empty channel unclarified)
+ NASA SMAP/MSL: Exactly 81 verified active channels (55 SMAP, 26 MSL)

- OPS-SAT Telemetry: Single 0.0665 average (masking 0-anomaly channel distortion)
+ OPS-SAT Telemetry: Dual-reported (0.0665 all-9 macro avg vs 0.0748 active-8 avg with per-channel provenance)

- ESA-ADB Sample: 0.8621 reported without sample-size warning
+ ESA-ADB Sample: 0.8621 reported with explicit low_sample_size_caveat=True (1 GT event)

- SMD Server Telemetry: 12-channel count ambiguously standing next to literature claims
+ SMD Server Telemetry: Formally reconciled (12-channel spot check vs authoritative 1,064-channel full set)

- SKAB / UCR: Loader stubs only
+ SKAB / UCR: Automated downloaders, parsers, and zero-leakage evaluation adapters fully implemented
```

## Recommended Plain-Language Paper Reporting Standards
1. **Never claim "perfect" or "clean" anomaly detection when event counts are under 10**. Always include the `low_sample_size_caveat` in table footnotes.
2. **Dual-report macro metrics** whenever benchmarks contain clean channels with zero anomalies.
3. **Report Raw-F1 as primary** and Affiliation-F1 as secondary; never lead with Point-Adjusted F1 (PA-F1) against unadjusted literature.

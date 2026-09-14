# End-to-End Generalization Pipeline Assessment

**Workspace:** `cubesat_project/full_audit_v1/`  
**Audit Question:** *Is there currently ONE single unified pipeline from raw telemetry -> preprocessing -> model -> threshold -> evaluation, runnable on any new unseen dataset without manual code changes? Or does every dataset require hand-written one-off loaders and scripts?*

---

## 1. Current State of the Codebase (Direct Finding)

**Finding:** The project currently relies on **separate, hand-written dataset loaders and evaluation scripts** across different benchmark datasets:
- **NASA SMAP/MSL:** Evaluated via `fresh_pipeline.py` and `run_master_integrity_pass.py` (expects `data/train/{chan}.npy`, `data/test/{chan}.npy`, and `labeled_anomalies.csv`).
- **ESA OPS-SAT:** Evaluated via `CubeSat_Anomaly_Detection__FINAL_ALL_CHECKPOINTS.ipynb` Phase 3 (expects `opssat_data/dataset.csv` and `segments.csv`).
- **ESA-ADB:** Evaluated via `ESA_ADB_Cross_Mission_Extension.ipynb` Phase 5 (expects `esa_adb_data/esa_adb_mission_telemetry.csv`).
- **SMD / NAB / External Benchmarks:** Evaluated via historical separate loader functions in `CubeSat_Results_Analysis.ipynb`.

### Crucial Methodological Distinction:
> **The *model architecture* generalizes across aerospace telemetry streams, but the *software pipeline* is not yet a single generalized end-to-end framework.**

This distinction must be stated plainly in any research paper draft:
1. The **MultiScale micro-architecture and distillation protocol** successfully transfer representations across datasets (demonstrated on NASA SMAP/MSL and ESA OPS-SAT).
2. However, ingesting a new mission dataset currently requires writing a custom schema parser (to extract timestamps, telemetry channels, and ground-truth event masks).

---

## 2. Specification for a True Unified Telemetry Pipeline (`TelemetryDataset` Interface)

To make the pipeline 100% end-to-end automated for any arbitrary satellite mission, the following unified adapter pattern is required:

```python
class BaseTelemetryAdapter(ABC):
    """Abstract Base Interface for any Spacecraft Telemetry Benchmark."""
    
    @abstractmethod
    def load_telemetry_stream(self, stream_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            train_raw: (N_train, n_channels) continuous & discrete telemetry
            test_raw:  (N_test, n_channels)
            test_labels: (N_test,) binary ground truth anomaly mask
        """
        pass
        
    @abstractmethod
    def list_streams(self) -> List[str]:
        """Returns all available channel/stream identifiers."""
        pass

class UnifiedTelemetryPipeline:
    """Single end-to-end runner that operates identically on any BaseTelemetryAdapter."""
    def __init__(self, adapter: BaseTelemetryAdapter, window_size: int = 32, stride: int = 4):
        self.adapter = adapter
        self.window_size = window_size
        self.stride = stride
        
    def run_benchmark(self, model_builder, scaler_type='standard') -> pd.DataFrame:
        results = []
        for stream_id in self.adapter.list_streams():
            tr, te, labels = self.adapter.load_telemetry_stream(stream_id)
            # 1. Zero-leakage train-fitted normalization
            # 2. Windowing & validation split
            # 3. Model training
            # 4. Out-of-sample threshold calibration (tau*)
            # 5. Inference & strict metric computation (Raw-F1, Aff-F1, PR-AUC)
            ...
        return pd.DataFrame(results)
```

---

## 3. Actionable Recommendation for Paper Framing
- **Do NOT claim:** *"We present a fully automated push-button pipeline that ingests any satellite data without code modification."*
- **DO claim:** *"We present a lightweight Multi-Scale edge anomaly detection architecture and distillation protocol that generalizes across NASA and ESA satellite telemetry, requiring only a lightweight schema adapter for mission-specific data ingestion."*

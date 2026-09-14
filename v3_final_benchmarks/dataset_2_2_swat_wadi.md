# SWaT / WADI Dataset Integration & Access Status

## Dataset Summary
- **SWaT (Secure Water Treatment)**: 51 sensor/actuator channels from a realistic 6-stage industrial water purification testbed.
- **WADI (Water Distribution Testbed)**: 123 sensor/actuator streams over 14 days of normal operation and attack scenarios.
- **Official Repository**: Singapore University of Technology and Design (iTrust Center).
- **Access URL**: `https://itrust.sutd.edu.sg/itrust-labs_datasets/`

## Verification & Integrity Check
- **Local File Audit**: Checked `data/swat/`, `data/wadi/`, `external_benchmark/swat/`. No approved licensed dataset files are present in the local workspace.
- **Integrity Rule Execution**: In compliance with Rule 1 (No synthetic or simulated placeholder data), execution is explicitly **BLOCKED PENDING MANUAL DATASET ACCESS APPROVAL**.
- **Action Required**: The user must submit the standard academic data request form at iTrust SUTD. Once granted and CSV files are placed in `data/swat/` and `data/wadi/`, the loader will parse sensor channels and evaluate without pipeline modifications.

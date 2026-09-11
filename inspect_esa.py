import pandas as pd
import numpy as np

df = pd.read_csv('esa_adb_data/esa_adb_mission_telemetry.csv')
print('=== ESA-ADB FILE DETAILED INSPECTION ===')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('\nHead:')
print(df.head(10))
print('\nSummary Statistics:')
print(df.describe())

print('\nTimestamps:')
print('Start:', df['timestamp'].iloc[0])
print('End:  ', df['timestamp'].iloc[-1])

anomaly_indices = np.where(df['is_anomaly'] == 1)[0]
print('\nTotal Anomaly Points:', len(anomaly_indices))

diffs = np.diff(anomaly_indices)
split_points = np.where(diffs > 1)[0] + 1
segments = np.split(anomaly_indices, split_points)
print(f'Total Contiguous Anomaly Events: {len(segments)}')
for idx, seg in enumerate(segments):
    s_idx, e_idx = seg[0], seg[-1]
    print(f'  Event {idx+1}: index [{s_idx} : {e_idx}] (len={len(seg)} rows) | ts: {df["timestamp"].iloc[s_idx]} to {df["timestamp"].iloc[e_idx]}')
    print(f'    Anomaly Power Mean: {df["power_telemetry"].iloc[seg].mean():.4f}, Normal Power Mean: {df.loc[df["is_anomaly"]==0, "power_telemetry"].mean():.4f}')
    print(f'    Anomaly Thermal Mean: {df["thermal_telemetry"].iloc[seg].mean():.4f}, Normal Thermal Mean: {df.loc[df["is_anomaly"]==0, "thermal_telemetry"].mean():.4f}')

# Check signal properties (autocorrelation, seasonality)
power = df['power_telemetry'].values
print('\nSignal Autocorrelation (lag 1, 10, 50, 100, 500):')
for lag in [1, 10, 50, 100, 500]:
    ac = np.corrcoef(power[:-lag], power[lag:])[0, 1]
    print(f'  Lag {lag:>3}: {ac:.4f}')

# Check timestamps step
ts = pd.to_datetime(df['timestamp'])
steps = ts.diff().dropna().unique()
print('\nTimestamp steps:', steps)

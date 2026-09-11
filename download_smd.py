import os
import urllib.request

smd_root = 'external_benchmark/data_external/smd'
test_dir = os.path.join(smd_root, 'test')
label_dir = os.path.join(smd_root, 'test_label')
os.makedirs(test_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# SMD machine list
machines = []
for group in range(1, 4):
    for num in range(1, 12):
        if group == 1 and num > 8:
            continue
        if group == 2 and num > 9:
            continue
        if group == 3 and num > 11:
            continue
        machines.append(f"machine-{group}-{num}")

base_url = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"

success_count = 0
for m in machines:
    test_url = f"{base_url}/test/{m}.txt"
    label_url = f"{base_url}/test_label/{m}.txt"
    
    test_dst = os.path.join(test_dir, f"{m}.txt")
    label_dst = os.path.join(label_dir, f"{m}.txt")
    
    try:
        urllib.request.urlretrieve(test_url, test_dst)
        urllib.request.urlretrieve(label_url, label_dst)
        success_count += 1
        print(f"[Downloaded SMD] {m}")
    except Exception as e:
        print(f"[Failed SMD] {m}: {e}")

print(f"Successfully downloaded {success_count}/{len(machines)} SMD machines.")

"""Paper 1 - collect M3/M2 outputs into single CSV files (CPU only, a few seconds, read-only except the 3 new CSVs)."""
from google.colab import drive
drive.mount('/content/drive')
import os, json, glob
import pandas as pd
M3 = '/content/drive/MyDrive/P1_m3'
M2 = '/content/drive/MyDrive/P1_m2'
rows, runs = [], []
for d in sorted(glob.glob(M3 + '/runs/*_seed*')):
    name = os.path.basename(d); model, seed = name.rsplit('_seed', 1)
    if not os.path.exists(d + '/run.json'):
        print('MISSING run.json', name); continue
    r = json.load(open(d + '/run.json'))
    runs.append({k: v for k, v in r.items() if not isinstance(v, dict)} |
                {f'{p}_{k}': v for p in ('es', 'final', 'min20') for k, v in r[p].items()})
    for p in ('es', 'final', 'min20'):
        t = pd.read_csv(f'{d}/{p}_per_image.csv'); t.insert(0, 'policy', p); t.insert(0, 'seed', int(seed)); t.insert(0, 'model', model)
        rows.append(t)
pd.concat(rows).to_csv(M3 + '/m3_all_per_image.csv', index=False)
pd.DataFrame(runs).to_csv(M3 + '/m3_runs.csv', index=False)
v = []
for d in sorted(glob.glob(M2 + '/runs/*_seed*')):
    if os.path.exists(d + '/val_severing.csv'):
        t = pd.read_csv(d + '/val_severing.csv'); t.insert(0, 'run', os.path.basename(d)); v.append(t)
if v:
    pd.concat(v).to_csv(M2 + '/m2_val_severing_all.csv', index=False)
print('runs collected:', len(runs), '| per-image rows:', sum(len(t) for t in rows), '| M2 files:', len(v))

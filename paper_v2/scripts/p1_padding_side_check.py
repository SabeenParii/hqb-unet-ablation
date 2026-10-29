"""
Paper 1 · Image-only check of the padded side (no training, read-only)

For every one of the 530 images, compare
  * mask rule  : side used in the paper = half of the full-resolution PROXY MASK with more foreground
                 (qseg.data_prep.chest_wall_side)
  * image rule : half of the full-resolution IMAGE with the higher mean gray value
If both rules pick the same side for all 530 images, the proxy mask did not influence preprocessing
beyond what the image alone determines.

READ ONLY: QSEG_FINAL (code, manifest, 530-images) is only read. Nothing is written to Drive;
the result (a small CSV with side labels, no pixels) stays on the Colab VM in /content/padding_check.
"""
from google.colab import drive
drive.mount('/content/drive')
import os, sys, glob
import numpy as np, pandas as pd, cv2

ROOT = '/content/drive/MyDrive/QSEG_FINAL'
sys.path.insert(0, f'{ROOT}/code')
from qseg.config import Config
from qseg.data_prep import generate_breast_mask, chest_wall_side

cfg = Config()
man = pd.read_csv(f'{ROOT}/results/data/manifest.csv')
print('images in manifest:', len(man))

# locate the raw files (searched recursively in 530-images; read only)
paths = {os.path.basename(p): p for p in glob.glob(f'{ROOT}/530-images/**/*', recursive=True)
         if p.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))}
missing = [f for f in man.original_filename if f not in paths]
print('raw files found:', len(paths), '| missing for manifest:', len(missing))
assert not missing, missing[:5]

rows = []
for i, r in man.iterrows():
    gray = cv2.imread(paths[r.original_filename], cv2.IMREAD_GRAYSCALE)
    mask, _, _ = generate_breast_mask(gray, cfg.otsu_kernel)
    side_mask = chest_wall_side(mask)                       # rule used in the paper
    w = gray.shape[1]
    side_img = 'right' if gray[:, w // 2:].mean() >= gray[:, :w // 2].mean() else 'left'
    rows.append(dict(file_id=r.file_id, split=r.split, side_mask=side_mask, side_image=side_img,
                     agree=side_mask == side_img))
    if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{len(man)}')

res = pd.DataFrame(rows)
os.makedirs('/content/padding_check', exist_ok=True)
res.to_csv('/content/padding_check/padding_side_check.csv', index=False)
print('\nAGREEMENT, all images :', int(res.agree.sum()), '/', len(res))
print(res.groupby('split').agree.agg(['sum', 'count']))
if not res.agree.all():
    print('\nDISAGREEING images (file_id, split, mask rule, image rule):')
    print(res[~res.agree][['file_id', 'split', 'side_mask', 'side_image']].to_string(index=False))

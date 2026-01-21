import sys
sys.path.insert(0, 'E:/USV/mmsegmentation')

import os
import cv2
import numpy as np
import mmcv
import torch

from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

register_all_modules()

# -----------------------------
# PATHS
# -----------------------------
CONFIG = 'E:/USV/mmsegmentation/configs/mask2former/mask2former_swin-l_lars_512x1024.py'
CHECKPOINT = 'E:/USV/mmsegmentation/work_dirs/mask2former_swin-l_lars_512x1024/best_mIoU_iter_6039.pth'

IMAGE_DIR = 'E:/USV/Dataset/lars_v1.0.0_images/val/images'
GT_DIR = 'E:/USV/Dataset/lars_v1.0.0_annotations/val/semantic_masks'
OUT_DIR = 'E:/USV/mmsegmentation/paper_resultsA'

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# PUBLICATION COLORS
# -----------------------------
PUB_CLASSES = ['obstacle', 'water', 'sky']
PUB_PALETTE = np.array([
    (220, 20, 60),     # obstacle
    (30, 144, 255),    # water
    (255, 215, 0)      # sky
], dtype=np.uint8)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = init_model(CONFIG, CHECKPOINT, device='cuda:0')
model.eval()

model.dataset_meta = dict(
    classes=PUB_CLASSES,
    palette=PUB_PALETTE.tolist()
)

# -----------------------------
# UTILS
# -----------------------------
def colorize(mask):
    return PUB_PALETTE[mask]

def overlay(img, mask, alpha=0.5):
    return (img * (1 - alpha) + mask * alpha).astype(np.uint8)

# -----------------------------
# MAIN (SAFE PATH)
# -----------------------------
for name in sorted(os.listdir(IMAGE_DIR)):
    if not name.endswith('.jpg'):
        continue

    img_path = os.path.join(IMAGE_DIR, name)
    gt_path = os.path.join(GT_DIR, name.replace('.jpg', '.png'))

    img = mmcv.imread(img_path)
    gt = mmcv.imread(gt_path, flag='grayscale')

    # Use the official helper which prepares data (to-tensor, batching, meta)
    # inference_model handles preprocessing and returns a SegDataSample
    # (or list[SegDataSample] for a batch). This avoids treating numpy
    # arrays as tensors (which caused the `.size()` call to fail).
    output = inference_model(model, img)

    # inference_model returns either a single SegDataSample or a list.
    res = output[0] if isinstance(output, (list, tuple)) else output
    pred = res.pred_sem_seg.data.cpu().numpy().squeeze()

    gt_color = colorize(gt)
    pred_color = colorize(pred)
    pred_overlay = overlay(img, pred_color)

    figure = np.hstack([img, gt_color, pred_color, pred_overlay])
    cv2.imwrite(os.path.join(OUT_DIR, name), figure)

print("✅ Publication-ready figures saved in:", OUT_DIR)

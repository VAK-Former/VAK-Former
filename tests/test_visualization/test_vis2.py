import sys
import cv2
import torch
import numpy as np

# ---------------------------------------------------------
# Make mmseg visible
# ---------------------------------------------------------
sys.path.insert(0, r"E:/USV/mmsegmentation")

from mmseg.apis import init_model
from mmseg.utils import register_all_modules
from mmseg.structures import SegDataSample

register_all_modules()

config = r"E:/USV/mmsegmentation/configs/mask2former/mask2former_swin-l_lars_512x1024.py"
checkpoint = r"E:/USV/mmsegmentation/work_dirs/mask2former_swin-l_lars_512x1024/best_mIoU_iter_6039.pth"
img_path = r"E:/USV/mmsegmentation/Final Image results/15.0.jpg"
out_path = "E:/USV/mmsegmentation/vis_img/20.2.png"

model = init_model(config, checkpoint, device="cuda:0")
model.eval()

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ori_h, ori_w = img.shape[:2]

target_w, target_h = 1024, 512
scale = min(target_w / ori_w, target_h / ori_h)
new_w, new_h = int(ori_w * scale), int(ori_h * scale)

img_resized = cv2.resize(img, (new_w, new_h))

mean = np.array([123.675, 116.28, 103.53])
std = np.array([58.395, 57.12, 57.375])
img_norm = (img_resized - mean) / std

pad_h = (32 - img_norm.shape[0] % 32) % 32
pad_w = (32 - img_norm.shape[1] % 32) % 32
img_pad = np.pad(img_norm, ((0, pad_h), (0, pad_w), (0, 0)))

inputs = torch.from_numpy(img_pad).permute(2, 0, 1).float().unsqueeze(0).cuda()

data_sample = SegDataSample()
data_sample.set_metainfo(dict(
    ori_shape=(ori_h, ori_w),
    img_shape=img_pad.shape[:2],
    pad_shape=img_pad.shape[:2],
    scale_factor=(scale, scale),
    flip=False
))

data = dict(
    inputs=inputs,
    data_samples=[data_sample]
)

with torch.no_grad():
    result = model.test_step(data)[0]

pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy()
pred = cv2.resize(pred, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

OBSTACLE_ID = 0

obstacle_mask = (pred == OBSTACLE_ID)
obstacle_mask = np.roll(obstacle_mask, shift=17, axis=1)
overlay = img.copy()
overlay[obstacle_mask] = [255, 255, 0]  # yellow obstacles

vis = (0.3 * img + 0.6 * overlay).astype(np.uint8)

cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

print("✅ Obstacle-only visualization saved:", out_path)
print("Unique labels in prediction:", np.unique(pred))

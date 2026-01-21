import sys
import torch
import cv2
import numpy as np
from mmseg.apis import init_model
from mmseg.utils import register_all_modules
from mmengine.structures import InstanceData
from mmseg.structures import SegDataSample

sys.path.insert(0, r"E:\USV\mmsegmentation")

register_all_modules()

config = r"E:/USV/mmsegmentation/configs/mask2former/mask2former_swin-l_lars_512x1024.py"
checkpoint = r"E:/USV/mmsegmentation/work_dirs/mask2former_swin-l_lars_512x1024/best_mIoU_iter_6039.pth"
img_path = r"E:/USV/Dataset/lars_v1.0.0_images/test/images/yt103_00_00379.jpg"

model = init_model(config, checkpoint, device="cuda:0")
model.eval()

print("CUDA available:", torch.cuda.is_available())
print("Model device:", next(model.parameters()).device)

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

palette = np.array(model.dataset_meta["palette"])
color_mask = palette[pred]
color_mask = np.roll(color_mask, shift=17, axis=1)
color_mask = cv2.resize(color_mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

overlay = (0.6 * img + 0.8 * color_mask).astype(np.uint8)


out_path = "output_gpu.png"
cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("✅ Mask2Former inference successful")
print(f"✅ Output saved as {out_path}")
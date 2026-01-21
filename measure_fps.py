import time
import torch
from mmengine.config import Config
from mmengine.runner import Runner

CFG = 'configs/mask2former/mask2former_swin-l_lars_512x1024.py'
CKPT = 'work_dirs/mask2former_swin-l_lars_512x1024/best_mIoU_iter_6039.pth'
WARMUP = 20
ITERS = 100

cfg = Config.fromfile(CFG)
cfg.work_dir = './tmp_fps'
cfg.load_from = CKPT
cfg.model.pretrained = None

runner = Runner.from_cfg(cfg)
model = runner.model.cuda().eval()

dummy = torch.randn(1, 3, 512, 1024).cuda()

def forward_only(x):
    feats = model.backbone(x)
    if hasattr(model, 'neck') and model.neck is not None:
        feats = model.neck(feats)
    if hasattr(model.decode_head, 'pixel_decoder'):
        feats = model.decode_head.pixel_decoder(feats)
    return feats

# Warm-up
with torch.no_grad():
    for _ in range(WARMUP):
        forward_only(dummy)

torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    for _ in range(ITERS):
        forward_only(dummy)

torch.cuda.synchronize()
end = time.time()

fps = ITERS / (end - start)
latency = (end - start) / ITERS * 1000

print(f"FPS (backbone+pixel decoder): {fps:.2f}")
print(f"Latency: {latency:.2f} ms")

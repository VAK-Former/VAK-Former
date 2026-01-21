import os
import numpy as np
from PIL import Image

# INPUT: Panoptic (RGB) masks folder
mask_dir = r"E:\USV\Dataset\lars_v1.0.0_annotations\val\panoptic_masks"

# OUTPUT: Grayscale label masks folder
output_dir = r"E:\USV\Dataset\lars_v1.0.0_annotations\val\panoptic_masks_gray"
os.makedirs(output_dir, exist_ok=True)

# COLOR → CLASS ID MAPPING (IMPORTANT)
palette_to_label = {
    (0, 0, 0): 0,        # obstacle / background
    (128, 0, 0): 1,      # water
    (0, 128, 0): 2,      # sky
    (255, 255, 255): 255 # ignore pixels (optional)
}

for file in os.listdir(mask_dir):
    inp = os.path.join(mask_dir, file)
    out = os.path.join(output_dir, file)

    mask = np.array(Image.open(inp))

    # Convert RGB mask → class-ID mask
    if mask.ndim == 3:
        label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color, label in palette_to_label.items():
            matches = (mask == color).all(axis=-1)
            label_mask[matches] = label
    else:
        label_mask = mask  # already single-channel

    Image.fromarray(label_mask).save(out)

print("✅ Conversion complete!")
print("➡ Output saved to:", output_dir)

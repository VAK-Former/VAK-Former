from pathlib import Path
from PIL import Image

root = Path(r"E:\USV\Dataset")
img_dir = root / 'lars_v1.0.0_images' / 'val' / 'images'
mask_dir = root / 'lars_v1.0.0_annotations' / 'val' / 'semantic_masks'

mismatches = []
missing_masks = []

for img_path in sorted(img_dir.glob('*.jpg')):
    mask_path = mask_dir / (img_path.stem + '.png')
    if not mask_path.exists():
        missing_masks.append(str(img_path))
        continue
    try:
        with Image.open(img_path) as im:
            img_size = im.size  # (W, H)
        with Image.open(mask_path) as m:
            mask_size = m.size
    except Exception as e:
        print(f'Failed reading {img_path} or {mask_path}: {e}')
        continue
    if img_size != mask_size:
        mismatches.append((str(img_path.name), img_size, mask_size))

print(f'Total images checked: {len(list(img_dir.glob("*.jpg")))}')
print(f'Missing masks: {len(missing_masks)}')
if missing_masks:
    print('Examples of missing masks:')
    print('\n'.join(missing_masks[:5]))

print(f'Mismatched sizes: {len(mismatches)}')
if mismatches:
    print('First 30 mismatches:')
    for i, (name, img_size, mask_size) in enumerate(mismatches[:30], 1):
        print(f'{i}. {name}: image={img_size}, mask={mask_size}')
else:
    print('No mismatches found.')

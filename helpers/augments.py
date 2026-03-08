import torchvision.transforms as T
import torchvision.transforms.functional as TF

def tta_variants(image_list):
    variants = [image_list]
    variants.append([[img.flip(-1) for img in imgs] for imgs in image_list])
    # 90° rotations
    for angle in [15, 345]:
        variants.append([[TF.rotate(img, angle) for img in imgs] for imgs in image_list])
    for factor in [0.85, 1.15]:
        variants.append([[TF.adjust_contrast(img, factor) for img in imgs] for imgs in image_list])
    return variants  # 6 variants total including original

def make_transform(augment=False, size=224):
    ops = [T.Grayscale(), T.Resize((size, size))]
    if augment:
        ops += [T.RandomHorizontalFlip(), 
                T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.85, 1.15)),
                T.RandomAdjustSharpness(2, p=0.3),
                T.RandomAutocontrast(p=0.4),
               ]
    ops += [T.ToTensor(), 
            T.Normalize(mean=[0.449], std=[0.226])
           ]
    if augment:
        ops += [T.RandomErasing(p=0.1)
               ]
    return T.Compose(ops)
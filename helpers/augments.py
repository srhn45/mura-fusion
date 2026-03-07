import torchvision.transforms as T

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
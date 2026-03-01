import torchvision.transforms as T

def make_transform(augment=False):
    ops = [T.Grayscale(), T.Resize((224, 224))]
    if augment:
        ops += [T.RandomHorizontalFlip(), 
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.RandomAutocontrast(p=0.3),
                T.RandomAdjustSharpness(2, p=0.3),
               ]
    ops += [T.ToTensor(), 
            T.Normalize(mean=[0.449], std=[0.226])
           ]
    
    if augment:
        ops += [T.RandomErasing(p=0.1)
               ]
    return T.Compose(ops)
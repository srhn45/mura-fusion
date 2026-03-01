import torch
from pathlib import Path

_REGISTRY = {}

def register(cls):
    """Decorator to register an architecture class by name."""
    _REGISTRY[cls.__name__] = cls
    return cls

def save_checkpoint(model, path, **config):
    """
    Save state dict + config dict together.
    config should contain everything needed to reconstruct the model.
    """
    torch.save({
        "config": config,
        "state_dict": model.state_dict()
    }, path)

def load_checkpoint(path, device="cpu"):
    """
    Reconstruct model from checkpoint
    Returns the model in eval mode
    """
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]

    backbone_cls_name = config["backbone_cls"]
    backbone_kwargs   = config["backbone_kwargs"]
    classifier_kwargs = config["classifier_kwargs"]

    if backbone_cls_name not in _REGISTRY:
        raise ValueError(f"Unknown backbone '{backbone_cls_name}'. Make sure to import it")

    backbone = _REGISTRY[backbone_cls_name](**backbone_kwargs)

    from architectures.classifier import Classifier
    model = Classifier(backbone, **classifier_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, config
import torch
import torch.nn as nn
from typing import List


class ProgressiveUnfreezer:
    """
    Unfreezes frozen layer groups one at a time at val loss plateau

    Args:
        optimizer
        layer_groups    : list of nn.Module or lists of nn.Parameter, ordered last-layer-first
        lr_scale        : newly unfrozen group gets lr * lr_scale
        patience        : how many epochs of no improvement before unfreezing
        min_delta       : minimum improvement to reset the patience counter
        verbose         : print a message when a group is unfrozen

    Usage:
        Build groups ordered from deepest → shallowest
        groups = [
            backbone.layer4,
            backbone.layer3,
            backbone.layer2,
            backbone.layer1,
        ]
        unfreezer = ProgressiveUnfreezer(optimizer, groups, lr_scale=0.1, patience=3)

        Inside training loop, after evaluate():
        unfreezer.step(val_loss)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        layer_groups: list,
        lr_scale: float = 0.1,
        patience: int = 3,
        min_delta: float = 1e-3,
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.lr_scale = lr_scale
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        # accept nn.Module, list of modules, or list of parameters
        self.groups = [self._to_params(g) for g in layer_groups]
        self.n_groups = len(self.groups)
        self.next_idx = 0 

        self._best_loss = float("inf")
        self._wait = 0 # epochs since last improvement

        # skip groups that are already unfrozen
        while self.next_idx < self.n_groups and self._group_is_unfrozen(self.next_idx):
            self.next_idx += 1

        if verbose:
            frozen = sum(1 for i in range(self.next_idx, self.n_groups))
            print(f"[Unfreezer] {frozen} layer group(s) queued for progressive unfreezing.")

            
    def step(self, val_loss: float) -> bool:
        """
        Call once per epoch after validation.
        Returns True if a group was unfrozen, False otherwise.
        """
        improved = val_loss < self._best_loss - self.min_delta
        if improved:
            self._best_loss = val_loss
            self._wait = 0
            return False

        self._wait += 1
        if self._wait >= self.patience:
            self._wait = 0
            return self._unfreeze_next()

        return False

    def all_unfrozen(self) -> bool:
        """True when every group has been unfrozen."""
        return self.next_idx >= self.n_groups

    def status(self) -> str:
        return (
            f"[Unfreezer] {self.next_idx}/{self.n_groups} groups unfrozen | "
            f"plateau patience: {self._wait}/{self.patience} | "
            f"best_loss: {self._best_loss:.4f}"
        )

    def _unfreeze_next(self) -> bool:
        if self.next_idx >= self.n_groups:
            if self.verbose:
                print("[Unfreezer] All groups already unfrozen.")
            return False

        params = self.groups[self.next_idx]
        for p in params:
            p.requires_grad = True

        # Add the newly unfrozen params as a new optimizer param group 
        # scaling down learning rate to not blast pretrained weights
        base_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = base_lr * self.lr_scale
        self.optimizer.add_param_group({"params": params, "lr": new_lr})

        if self.verbose:
            n_params = sum(p.numel() for p in params)
            print(
                f"[Unfreezer] Unfroze group {self.next_idx + 1}/{self.n_groups} "
                f"({n_params:,} params) → lr={new_lr:.2e}"
            )

        self.next_idx += 1
        return True

    def _group_is_unfrozen(self, idx: int) -> bool:
        return all(p.requires_grad for p in self.groups[idx])

    @staticmethod
    def _to_params(group) -> list:
        """Normalize a group into a flat list of nn.Parameters."""
        if isinstance(group, nn.Module):
            return list(group.parameters())
        if isinstance(group, (list, tuple)):
            params = []
            for item in group:
                if isinstance(item, nn.Module):
                    params.extend(item.parameters())
                elif isinstance(item, nn.Parameter):
                    params.append(item)
                else:
                    raise TypeError(f"Unexpected type in group: {type(item)}")
            return params
        if isinstance(group, nn.Parameter):
            return [group]
        raise TypeError(f"Cannot convert {type(group)} to parameter list.")



#  Helper: auto-extract groups from a Sequential or ModuleList

def groups_from_sequential(module: nn.Module, reverse: bool = True) -> list:
    """
    Extracts direct children of a Sequential or ModuleList as individual
    groups, ordered last-first (reverse=True) ready for ProgressiveUnfreezer.

    """
    children = list(module.children())
    if reverse:
        children = list(reversed(children))
    return children
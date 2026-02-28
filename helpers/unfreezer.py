import torch
import torch.nn as nn
from typing import List


class ProgressiveUnfreezer:
    """Unfreezes layer groups progressively based on a monitored metric."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        layer_groups: list,
        lr_scale: float = 0.1,
        patience: int = 3,
        min_delta: float = 1e-3,
        mode: str = "min",  # "min" for val_loss, "max" for kappa/accuracy
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.lr_scale = lr_scale
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.groups = [self._to_params(g) for g in layer_groups]
        self.n_groups = len(self.groups)
        self.next_idx = 0

        self._best_val = float("inf") if mode == "min" else -float("inf")
        self._wait = 0

        while self.next_idx < self.n_groups and self._group_is_unfrozen(self.next_idx):
            self.next_idx += 1

        if verbose:
            frozen = self.n_groups - self.next_idx
            print(f"[Unfreezer] {frozen} group(s) queued for progressive unfreezing.")

    def step(self, metric: float) -> bool:
        improved = (metric < self._best_val - self.min_delta) if self.mode == "min" else (metric > self._best_val + self.min_delta)
        if improved:
            self._best_val = metric
            self._wait = 0
            return False

        self._wait += 1
        if self._wait >= self.patience:
            self._wait = 0
            return self._unfreeze_next()
        return False

    def all_unfrozen(self) -> bool:
        return self.next_idx >= self.n_groups

    def status(self) -> str:
        return f"[Unfreezer] {self.next_idx}/{self.n_groups} groups unfrozen | wait: {self._wait}/{self.patience} | best_val: {self._best_val:.4f}"

    def _unfreeze_next(self) -> bool:
        if self.next_idx >= self.n_groups:
            if self.verbose:
                print("[Unfreezer] All groups already unfrozen.")
            return False

        params = self.groups[self.next_idx]
        for p in params:
            p.requires_grad = True

        base_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.add_param_group({"params": params, "lr": base_lr * self.lr_scale})

        if self.verbose:
            n_params = sum(p.numel() for p in params)
            print(f"[Unfreezer] Unfroze group {self.next_idx+1}/{self.n_groups} ({n_params:,} params) → lr={base_lr*self.lr_scale:.2e}")

        self.next_idx += 1
        return True

    def _group_is_unfrozen(self, idx: int) -> bool:
        return all(p.requires_grad for p in self.groups[idx])

    @staticmethod
    def _to_params(group) -> list:
        if isinstance(group, nn.Module): return list(group.parameters())
        if isinstance(group, (list, tuple)):
            params = []
            for g in group:
                if isinstance(g, nn.Module): params.extend(g.parameters())
                elif isinstance(g, nn.Parameter): params.append(g)
                else: raise TypeError(type(g))
            return params
        if isinstance(group, nn.Parameter): return [group]
        raise TypeError(type(group))



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
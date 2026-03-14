import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

CATEGORIES_SHORT = ["SHOULDER", "HUMERUS", "ELBOW", "FOREARM", "WRIST", "HAND", "FINGER"]
CAT_COLORS = ["#60a5fa", "#34d399", "#f472b6", "#fb923c", "#a78bfa", "#facc15", "#f87171"]

class Reporter:
    def __init__(self, checkpoint_path, model_name, backbone_kwargs, classifier_kwargs, fit_kwargs):
        self.base     = checkpoint_path.replace(".pt", "")
        self.metadata = {
            "model_name":        model_name,
            "backbone_kwargs":   backbone_kwargs,
            "classifier_kwargs": classifier_kwargs,
            "fit_kwargs":        {k: v for k, v in fit_kwargs.items()
                                  if not isinstance(v, dict)},  # skip pos_weight dict
            "timestamp":         datetime.now().isoformat(),
            "platform":          platform.node(),
        }
        # per-epoch lists
        self.epochs        = []
        self.train_loss    = []
        self.train_acc     = []
        self.val_loss      = []
        self.val_acc       = []
        self.kappa         = []
        self.tuned_kappa   = []
        self.tuned_thresh  = []
        self.cat_kappa     = {c: [] for c in CATEGORIES_SHORT}   # per-category kappa
        self.cat_acc       = {c: [] for c in CATEGORIES_SHORT}
        self.best_epoch    = None
        self.best_kappa    = -1.0

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc,
                  kappa, tuned_kappa, tuned_thresh, per_cat):
        """
        per_cat: dict { "XR_SHOULDER": (kappa, acc), ... }
        """
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.kappa.append(kappa)
        self.tuned_kappa.append(tuned_kappa)
        self.tuned_thresh.append(tuned_thresh)

        for short in CATEGORIES_SHORT:
            key = f"XR_{short}"
            if key in per_cat:
                self.cat_kappa[short].append(per_cat[key][0])
                self.cat_acc[short].append(per_cat[key][1])
            else:
                self.cat_kappa[short].append(None)
                self.cat_acc[short].append(None)

        if kappa > self.best_kappa:
            self.best_kappa = kappa
            self.best_epoch = epoch

    def save(self):
        self._save_plots()
        self._save_report()
        self._save_json()

    # ── Plots ──────────────────────────────────────────────────────────────────

    def _save_plots(self):
        ep = self.epochs
        fig = plt.figure(figsize=(20, 14), facecolor="#0f1117")
        fig.suptitle(f"{self.metadata['model_name']}  —  Training Curves",
                     color="#e2e8f0", fontsize=15, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        style = dict(facecolor="#1e293b", tick_params=dict(colors="#64748b"),
                     label_color="#94a3b8", grid_color="#1e3a5f")

        def styled_ax(ax, title, ylabel=None):
            ax.set_facecolor(style["facecolor"])
            ax.tick_params(colors=style["tick_params"]["colors"], labelsize=8)
            ax.set_title(title, color="#cbd5e1", fontsize=10, pad=6)
            if ylabel:
                ax.set_ylabel(ylabel, color=style["label_color"], fontsize=8)
            ax.set_xlabel("Epoch", color=style["label_color"], fontsize=8)
            ax.grid(True, color=style["grid_color"], linewidth=0.5, alpha=0.6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")

        # 1. Loss
        ax = fig.add_subplot(gs[0, 0])
        styled_ax(ax, "Loss", "Loss")
        ax.plot(ep, self.train_loss, color="#60a5fa", linewidth=1.5, label="train")
        ax.plot(ep, self.val_loss,   color="#f472b6", linewidth=1.5, label="val")
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

        # 2. Accuracy
        ax = fig.add_subplot(gs[0, 1])
        styled_ax(ax, "Accuracy", "Accuracy")
        ax.plot(ep, self.train_acc, color="#60a5fa", linewidth=1.5, label="train")
        ax.plot(ep, self.val_acc,   color="#f472b6", linewidth=1.5, label="val")
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

        # 3. Global kappa
        ax = fig.add_subplot(gs[0, 2])
        styled_ax(ax, "Global κ", "κ")
        ax.plot(ep, self.kappa,       color="#34d399", linewidth=1.5, label="untuned")
        ax.plot(ep, self.tuned_kappa, color="#a78bfa", linewidth=1.5, label="tuned", linestyle="--")
        if self.best_epoch:
            ax.axvline(self.best_epoch, color="#fbbf24", linewidth=1, linestyle=":", alpha=0.8)
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

        # 4. Threshold over time
        ax = fig.add_subplot(gs[1, 0])
        styled_ax(ax, "Tuned Threshold", "Threshold")
        ax.plot(ep, self.tuned_thresh, color="#fb923c", linewidth=1.5)
        ax.axhline(0.5, color="#475569", linewidth=1, linestyle="--", alpha=0.6)

        # 5. Train/val acc gap (overfitting indicator)
        ax = fig.add_subplot(gs[1, 1])
        styled_ax(ax, "Train−Val Acc Gap", "Gap")
        gap = [t - v for t, v in zip(self.train_acc, self.val_acc)]
        ax.fill_between(ep, gap, alpha=0.3, color="#f87171")
        ax.plot(ep, gap, color="#f87171", linewidth=1.5)
        ax.axhline(0, color="#475569", linewidth=1, linestyle="--")

        # 6. Per-category kappa (line chart)
        ax = fig.add_subplot(gs[1, 2])
        styled_ax(ax, "Per-Category κ", "κ")
        for short, color in zip(CATEGORIES_SHORT, CAT_COLORS):
            vals = self.cat_kappa[short]
            clean = [v if v is not None else float("nan") for v in vals]
            ax.plot(ep, clean, color=color, linewidth=1.2, label=short, alpha=0.9)
        ax.legend(fontsize=6.5, facecolor="#1e293b", edgecolor="#334155",
                  labelcolor="#e2e8f0", ncol=2)

        # 7–9. Per-category kappa bar chart at best epoch
        best_i = self.epochs.index(self.best_epoch) if self.best_epoch else -1
        ax = fig.add_subplot(gs[2, :2])
        styled_ax(ax, f"Per-Category κ @ Best Epoch ({self.best_epoch})", "κ")
        kappas = [self.cat_kappa[c][best_i] or 0 for c in CATEGORIES_SHORT]
        bars = ax.bar(CATEGORIES_SHORT, kappas, color=CAT_COLORS, alpha=0.85, edgecolor="#334155")
        for bar, val in zip(bars, kappas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", color="#e2e8f0", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", colors="#94a3b8", labelsize=8)

        # 9. Val loss + kappa dual axis
        ax  = fig.add_subplot(gs[2, 2])
        ax2 = ax.twinx()
        styled_ax(ax, "Val Loss vs κ")
        ax.plot(ep, self.val_loss, color="#f472b6", linewidth=1.5)
        ax.set_ylabel("Val Loss", color="#f472b6", fontsize=8)
        ax.tick_params(axis="y", colors="#f472b6", labelsize=8)
        ax2.plot(ep, self.kappa, color="#34d399", linewidth=1.5, linestyle="--")
        ax2.set_ylabel("κ", color="#34d399", fontsize=8)
        ax2.tick_params(axis="y", colors="#34d399", labelsize=8)
        ax2.set_facecolor("#1e293b")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#334155")

        plt.savefig(f"{self.base}_curves.png", dpi=150, bbox_inches="tight",
                    facecolor="#0f1117")
        plt.close()
        print(f"  ↳ plots saved to {self.base}_curves.png")

    # ── Text report ────────────────────────────────────────────────────────────

    def _save_report(self):
        W = 64
        lines = []
        def w(s=""): lines.append(s)
        def hr(c="─"): w(c * W)
        def section(t): hr(); w(f"  {t}"); hr()

        best_i = self.epochs.index(self.best_epoch) if self.best_epoch else -1

        w("█" * W)
        w(f"{'ABLATION REPORT':^{W}}")
        w("█" * W)
        w(f"  Model:      {self.metadata['model_name']}")
        w(f"  Timestamp:  {self.metadata['timestamp']}")
        w(f"  Host:       {self.metadata['platform']}")

        section("CONFIG")
        for k, v in self.metadata["backbone_kwargs"].items():
            w(f"  backbone.{k:<25} {v}")
        for k, v in self.metadata["fit_kwargs"].items():
            w(f"  fit.{k:<29} {v}")

        section("BEST EPOCH SUMMARY")
        w(f"  Best epoch:     {self.best_epoch}")
        w(f"  Val kappa:      {self.kappa[best_i]:.4f}")
        w(f"  Tuned kappa:    {self.tuned_kappa[best_i]:.4f}  (t={self.tuned_thresh[best_i]:.2f})")
        w(f"  Val accuracy:   {self.val_acc[best_i]:.4f}")
        w(f"  Val loss:       {self.val_loss[best_i]:.4f}")
        w(f"  Train accuracy: {self.train_acc[best_i]:.4f}")
        w(f"  Train loss:     {self.train_loss[best_i]:.4f}")
        gap = self.train_acc[best_i] - self.val_acc[best_i]
        w(f"  Train-val gap:  {gap:.4f}")

        section("PER-CATEGORY @ BEST EPOCH")
        w(f"  {'Category':<12} {'κ':>8}  {'Acc':>8}")
        hr("·")
        for short in CATEGORIES_SHORT:
            k = self.cat_kappa[short][best_i]
            a = self.cat_acc[short][best_i]
            ks = f"{k:.4f}" if k is not None else "  N/A"
            as_ = f"{a:.4f}" if a is not None else "  N/A"
            w(f"  {short:<12} {ks:>8}  {as_:>8}")

        section("EPOCH-BY-EPOCH")
        w(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'VlLoss':>8}  {'VlAcc':>7}  {'κ':>7}  {'κ*':>7}  {'t*':>5}")
        hr("·")
        for i, ep in enumerate(self.epochs):
            marker = " ◀" if ep == self.best_epoch else ""
            w(f"  {ep:>3}  {self.train_loss[i]:>8.4f}  {self.train_acc[i]:>7.4f}"
              f"  {self.val_loss[i]:>8.4f}  {self.val_acc[i]:>7.4f}"
              f"  {self.kappa[i]:>7.4f}  {self.tuned_kappa[i]:>7.4f}"
              f"  {self.tuned_thresh[i]:>5.2f}{marker}")

        hr("█")
        w(f"{'END OF REPORT':^{W}}")
        hr("█")

        path = f"{self.base}_report.txt"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  ↳ report saved to {path}")

    # ── JSON (for programmatic comparison across ablations) ───────────────────

    def _save_json(self):
        data = {
            "metadata":     self.metadata,
            "best_epoch":   self.best_epoch,
            "best_kappa":   self.best_kappa,
            "epochs":       self.epochs,
            "train_loss":   self.train_loss,
            "train_acc":    self.train_acc,
            "val_loss":     self.val_loss,
            "val_acc":      self.val_acc,
            "kappa":        self.kappa,
            "tuned_kappa":  self.tuned_kappa,
            "tuned_thresh": self.tuned_thresh,
            "cat_kappa":    self.cat_kappa,
            "cat_acc":      self.cat_acc,
        }
        path = f"{self.base}_metrics.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ↳ metrics saved to {path}")
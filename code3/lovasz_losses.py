"""
lovasz_losses.py — Lovász-Softmax Loss for Semantic Segmentation
================================================================
Maximizes mIoU directly — much better than cross-entropy for
highly imbalanced multi-class segmentation.

Reference:
  Berman et al., "The Lovász-Softmax loss: A tractable surrogate for
  the optimization of the intersection-over-union measure in neural
  networks" (CVPR 2018)

Usage:
    from lovasz_losses import LovaszSoftmax
    criterion = LovaszSoftmax(ignore_index=255)
    loss = criterion(logits, targets)   # logits: (B,C,...) targets: (B,...)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


# ──────────────────────────────────────────────
# Core Lovász helpers
# ──────────────────────────────────────────────

def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovász-Softmax loss.
    probas : (P, C) float — class probabilities
    labels : (P,)   long  — GT class indices
    classes: 'all' | 'present' | list of class indices
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.shape[1]
    losses = []

    if classes == 'all':
        class_iter = range(C)
    elif classes == 'present':
        class_iter = labels.unique()
    else:
        class_iter = classes

    for c in class_iter:
        fg = (labels == c).float()          # foreground indicator
        if fg.sum() == 0 and classes == 'present':
            continue
        errors = (fg - probas[:, c]).abs()  # 1 - prob when fg, prob when bg
        errors_sorted, perm = torch.sort(errors, descending=True)
        perm = perm.detach()
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted,
                                lovasz_grad(fg_sorted).to(errors.device)))

    return torch.stack(losses).mean() if losses else probas.sum() * 0.0


def flatten_probas(probas, labels, ignore_index=None):
    """Flatten spatial dimensions."""
    if probas.dim() == 3:                    # (B, C, N)
        B, C, N = probas.shape
        probas = probas.permute(0, 2, 1).reshape(-1, C)
        labels = labels.reshape(-1)
    elif probas.dim() == 4:                  # (B, C, H, W)
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
        labels = labels.reshape(-1)
    elif probas.dim() == 5:                  # (B, C, D, H, W)
        B, C, D, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 4, 1).reshape(-1, C)
        labels = labels.reshape(-1)

    if ignore_index is not None:
        mask = labels != ignore_index
        probas = probas[mask]
        labels = labels[mask]

    return probas, labels


# ──────────────────────────────────────────────
# Main loss class
# ──────────────────────────────────────────────

class LovaszSoftmax(nn.Module):
    """
    Lovász-Softmax Loss + Weighted Cross-Entropy (combined).

    Parameters
    ----------
    classes      : 'present' → only classes present in GT
    per_image    : if True compute loss per image then average
    ignore_index : class index to ignore
    weights      : per-class weight tensor (for CE component)
    alpha        : weight of Lovász term  (1-alpha = CE weight)
    """
    def __init__(self,
                 classes='present',
                 per_image=False,
                 ignore_index=None,
                 weights=None,
                 alpha=0.7):
        super().__init__()
        self.classes      = classes
        self.per_image    = per_image
        self.ignore_index = ignore_index
        self.register_buffer('weights', weights)
        self.alpha        = alpha

    def forward(self, logits, targets):
        """
        logits  : (B, C, *spatial)  raw logits
        targets : (B, *spatial)     int64 GT labels
        """
        probas = F.softmax(logits, dim=1)

        # ── Lovász component ──
        if self.per_image:
            loss_lv = torch.stack([
                lovasz_softmax_flat(
                    *flatten_probas(p.unsqueeze(0), t.unsqueeze(0),
                                    self.ignore_index),
                    classes=self.classes)
                for p, t in zip(probas, targets)
            ]).mean()
        else:
            flat_p, flat_t = flatten_probas(probas, targets, self.ignore_index)
            loss_lv = lovasz_softmax_flat(flat_p, flat_t, self.classes)

        # ── Cross-Entropy component ──
        w = self.weights.to(logits.device) if self.weights is not None else None
        if self.ignore_index is not None:
            loss_ce = F.cross_entropy(logits, targets,
                                       weight=w,
                                       ignore_index=self.ignore_index)
        else:
            loss_ce = F.cross_entropy(logits, targets, weight=w)

        return self.alpha * loss_lv + (1.0 - self.alpha) * loss_ce


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == '__main__':
    B, C, D, H, W = 1, 5, 16, 20, 20
    logits  = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))
    crit = LovaszSoftmax(alpha=0.7)
    loss = crit(logits, targets)
    print(f'LovászSoftmax loss: {loss.item():.4f}  ✅')

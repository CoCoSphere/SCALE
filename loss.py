# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class PairLoss(nn.Module):
    """Binary pair cross-entropy loss with class weight computation utilities."""
    def __init__(self, class_weight=None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, logits, labels):
        return self.criterion(logits, labels.long())

    @staticmethod
    def compute_class_weights(train_loader, device, weight_ratio_cap=4.0):
        label_counts = torch.zeros(2)
        total_samples = 0
        for batch in train_loader:
            labels = batch['pair_labels']
            label_counts[0] += (labels == 0).sum().item()
            label_counts[1] += (labels == 1).sum().item()
            total_samples += labels.numel()
        if total_samples == 0 or label_counts.sum() == 0:
            return torch.ones(2, device=device)
        raw_weights = total_samples / (2 * label_counts)
        weight_ratio = raw_weights[1] / raw_weights[0]
        if weight_ratio > weight_ratio_cap:
            raw_weights[1] = raw_weights[0] * weight_ratio_cap
        return raw_weights.to(device)

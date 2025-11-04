# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import time
import os
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict

import logging
import sys
import copy


def set_seed(seed=42):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_time():
    """Print current time"""
    print(f'\n----------{time.strftime("%Y-%m-%d %X", time.localtime())}----------')
def _ensure_utf8_stdio():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def setup_logging(log_dir, dataset_name, model_type):

    import os
    log_file = os.path.join(log_dir, f'{dataset_name}_{model_type}.log')

    _ensure_utf8_stdio()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)



def list_round(a_list, decimals=4):
    return [round(float(i), decimals) for i in a_list]


class EarlyStopping:

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = copy.deepcopy(model.state_dict())


class MetricsCalculator:

    @staticmethod
    def calculate_prf(pred, true, mask=None, average='binary', use_emocate=False):

        if len(pred.shape) > 1:
            # Sequence labeling task
            pred_flat = []
            true_flat = []

            for i in range(pred.shape[0]):
                if mask is not None:
                    length = mask[i].sum().item()
                    pred_flat.extend(pred[i][:length].tolist())
                    true_flat.extend(true[i][:length].tolist())
                else:
                    pred_flat.extend(pred[i].tolist())
                    true_flat.extend(true[i].tolist())
        else:
            # Classification task
            pred_flat = pred.tolist()
            true_flat = true.tolist()

        # Auto-detect task type
        unique_labels = set(true_flat)
        n_classes = len(unique_labels)


        if use_emocate:
            if average == 'binary':
                average = 'macro' 
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_flat, pred_flat, average=average, zero_division=0
            )
        else:
            # Binary classification mode: neutral vs non-neutral
            if n_classes == 2 and average == 'binary':
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_flat, pred_flat, average='binary', zero_division=0
                )
            elif n_classes > 2 and average == 'binary':
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_flat, pred_flat, average='macro', zero_division=0
                )
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_flat, pred_flat, average=average, zero_division=0
                )

        return precision, recall, f1

    @staticmethod
    def calculate_emotion_category_f1(pred, true, mask=None, emotion_names=None):
        pred_flat = []
        true_flat = []

        for i in range(pred.shape[0]):
            if mask is not None:
                length = mask[i].sum().item()
                pred_flat.extend(pred[i][:length].tolist())
                true_flat.extend(true[i][:length].tolist())
            else:
                pred_flat.extend(pred[i].tolist())
                true_flat.extend(true[i].tolist())

        # Compute confusion matrix
        cm = confusion_matrix(true_flat, pred_flat)

        # Compute P, R, F1 for each category
        n_classes = cm.shape[0]
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision[i] = tp / (tp + fp + 1e-8)
            recall[i] = tp / (tp + fn + 1e-8)
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)

        # Compute weighted average (excluding neutral emotion category 0)
        if n_classes > 1:
            weights = cm[1:, :].sum(axis=1) 
            total_weight = weights.sum()
            if total_weight > 0:
                weighted_f1 = np.sum(f1[1:] * weights) / total_weight
            else:
                weighted_f1 = 0.0
        else:
            weighted_f1 = f1[0]

        results = {
            'per_class_f1': f1,
            'weighted_f1': weighted_f1,
            'macro_f1': f1[1:].mean() if n_classes > 1 else f1[0],  # Exclude neutral
            'confusion_matrix': cm
        }

        if emotion_names:
            results['emotion_names'] = emotion_names
            for i, name in enumerate(emotion_names):
                results[f'{name}_f1'] = f1[i]

        return results


class PairEvaluator:
    """End-to-end pair extraction evaluator"""

    @staticmethod
    def evaluate_pairs(all_true_pairs, all_pred_pairs):

        # Flatten all pairs
        true_pairs_flat = []
        pred_pairs_flat = []

        for true_pairs, pred_pairs in zip(all_true_pairs, all_pred_pairs):
            true_pairs_flat.extend(true_pairs)
            pred_pairs_flat.extend(pred_pairs)

        # Convert to sets
        true_set = set(true_pairs_flat)
        pred_set = set(pred_pairs_flat)

        # Compute metrics
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_true_pairs': len(true_pairs_flat),
            'n_pred_pairs': len(pred_pairs_flat)
        }

    @staticmethod
    def extract_pairs_from_predictions(conversation_ids, emo_ids, cause_ids,
                                     predictions, threshold=0.5):

        if predictions.dim() == 2:
            # Probability prediction
            pred_labels = (predictions[:, 1] > threshold).int()
        else:
            # Already labels
            pred_labels = predictions

        pairs = []
        for i, pred in enumerate(pred_labels):
            if pred == 1:
                pairs.append((conversation_ids[i], emo_ids[i].item(), cause_ids[i].item()))

        return pairs


class ModelSaver:

    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        # Save current checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints"""
        checkpoint_files = []
        for file in os.listdir(self.save_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
                checkpoint_files.append(file)

        if len(checkpoint_files) > self.max_checkpoints:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

            for file in checkpoint_files[:-self.max_checkpoints]:
                os.remove(os.path.join(self.save_dir, file))

    def load_checkpoint(self, model, optimizer=None, filename='best_model.pt'):
        """Load checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'], checkpoint['metrics']
        else:
            print(f"Checkpoint file {checkpoint_path} does not exist")
            return None, None
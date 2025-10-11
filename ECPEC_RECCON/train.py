# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm
import logging
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
from argparse import Namespace

# ??????
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_model_config
from data_loader import create_feature_data_loaders
from models import Model as MECPE_Model, PairLoss as PairLoss, MaskGenerator
from utils import (
    set_seed, print_time, MetricsCalculator, PairEvaluator,
    EarlyStopping, ModelSaver
)

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


def find_optimal_threshold(model, dataloader, device, logger, args, threshold_range=(0.3, 0.7), num_thresholds=21):
    results = evaluate(model, dataloader, device, logger, args=args, threshold=None)
    pair_probs = np.array(results['pair_probs'])
    pair_labels = np.array(results['pair_labels'])
    convo_ids = results['convo_ids']
    emo_ids = results['emo_ids']
    cause_ids = results['cause_ids']

    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    best_threshold = 0.5
    best_f1 = 0.0

    logger.info(f"Threshold search range: {threshold_range}, steps: {num_thresholds}")

    true_pairs_by_conv_full = getattr(dataloader.dataset, 'true_pairs_by_conv_full', {})

    for threshold in thresholds:
        preds = (pair_probs >= threshold).astype(int)
        pred_pairs_by_conv = defaultdict(list)

        for conv_id, emo_idx, cause_idx, pred in zip(convo_ids, emo_ids, cause_ids, preds):
            if pred == 1:
                pred_pairs_by_conv[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))

        all_conv_ids_full = set(true_pairs_by_conv_full.keys()) | set(pred_pairs_by_conv.keys())
        all_true_pairs_full = [true_pairs_by_conv_full.get(conv_id, []) for conv_id in all_conv_ids_full]
        all_pred_pairs_full = [pred_pairs_by_conv.get(conv_id, []) for conv_id in all_conv_ids_full]

        if all_conv_ids_full:
            pair_metrics = PairEvaluator.evaluate_pairs(all_true_pairs_full, all_pred_pairs_full)
            f1 = pair_metrics['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    logger.info(f"最佳阈值: {best_threshold:.3f}, 验证F1: {best_f1:.4f}")
    return best_threshold


def self_supervised_ot_loss(ot_scores, pair_indices, pair_labels, pair_conversation_id, threshold=0.0):
    device = ot_scores.device
    conv_to_rows = {}
    for row_i, conv_id in enumerate(pair_conversation_id):
        conv_to_rows.setdefault(conv_id, []).append(row_i)

    losses = []
    for conv_id, rows in conv_to_rows.items():
        rows_t = torch.tensor(rows, device=device, dtype=torch.long)
        emo_ids = pair_indices[rows_t, 1]
        scores = ot_scores[rows_t]
        labels = pair_labels[rows_t]

        emo_to_rows = {}
        for idx, e in enumerate(emo_ids.tolist()):
            emo_to_rows.setdefault(e, []).append(idx)

        for e_rows in emo_to_rows.values():
            e_rows_t = torch.tensor(e_rows, device=device, dtype=torch.long)
            local_scores = scores[e_rows_t]
            local_labels = labels[e_rows_t]

            if local_labels.sum() > 0:
                target = local_labels.float()
            else:
                max_idx = torch.argmax(local_scores)
                target = torch.zeros_like(local_scores)
                if threshold > 0:
                    mask = local_scores >= threshold
                    if mask.any():
                        target = mask.float()
                    else:
                        target[max_idx] = 1.0
                else:
                    target[max_idx] = 1.0

            pred = local_scores / (local_scores.max() + 1e-8)
            loss = F.binary_cross_entropy(pred, target)
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()



def compute_local_fgw_loss_and_probs(model, outputs, batch, device, args, mode='train'):
    """????????????????FGW???????CE???pair???
    - ???????: W_e = args.fgw_local_emo_window
    - ?????: ???? D = train_max_pair_distance/eval_max_pair_distance???? pred_future_cause ??
    """
    assert hasattr(model, 'ot_head') and model.ot_head is not None, 'FGW-only ????FGW? (use_ot_head=True)'

    emotion_context = outputs['emotion_context']
    cause_context = outputs['cause_context']
    edge_weights = outputs['edge_weights']
    doc_len = batch['doc_len'].to(device)

    pair_indices = batch['pair_indices'].to(device)
    pair_labels = batch['pair_labels'].to(device)
    convo_ids = batch['pair_conversation_id']

    # ????????per-conv set((e,c))
    pos_lookup = defaultdict(set)
    for (conv_id, e, c), lab in zip(pair_indices.tolist(), pair_labels.tolist()):
        if lab == 1:
            pos_lookup[int(conv_id)].add((int(e), int(c)))

    # ???????
    W_e = int(getattr(args, 'fgw_local_emo_window', 5))
    if mode == 'train':
        D = getattr(args, 'train_max_pair_distance', None)
    else:
        D = getattr(args, 'eval_max_pair_distance', None)
    # ?????????????????????????
    if D is None:
        D = 5
    else:
        D = int(D)
    allow_future = getattr(args, 'pred_future_cause', True)

    # ???????? pair_indices ?????
    prob_map = {}

    loss_terms = []
    row_count = 0

    for b in range(emotion_context.size(0)):
        L = int(doc_len[b].item())
        if L <= 0:
            continue
        A_full = edge_weights[b]

        for i in range(L):  # ?????
            # ???????
            e_start = max(0, i - W_e)
            e_end = min(L - 1, i + W_e)
            E_idx = torch.arange(e_start, e_end + 1, device=device, dtype=torch.long)

            # ??????????+???
            if allow_future:
                c_left = 0 if D is None else max(0, i - D)
                c_right = L - 1 if D is None else min(L - 1, i + D)
            else:
                c_left = 0 if D is None else max(0, i - D)
                c_right = i if D is None else min(i, i + D)
            if c_left > c_right:
                continue
            C_idx = torch.arange(c_left, c_right + 1, device=device, dtype=torch.long)

            # ????/??
            E_feat = emotion_context[b, E_idx]
            C_feat = cause_context[b, C_idx]
            A_e = A_full.index_select(0, E_idx).index_select(1, E_idx)
            A_c = A_full.index_select(0, C_idx).index_select(1, C_idx)

            # ??????
            T = model.ot_head._fgw_single(E_feat, C_feat, A_e, A_c, allowed_mask=None)
            # ???????
            if model.ot_head.row_norm == 'row_softmax':
                logits = T / max(model.ot_head.row_temp, 1e-6)
                T_norm = F.softmax(logits, dim=1)
            elif model.ot_head.row_norm == 'max':
                row_max = T.max(dim=1, keepdim=True).values.clamp_min(1e-8)
                T_norm = T / row_max
            else:
                T_norm = T

            # ?????????(k)???
            center_row = int(i - e_start)
            row_probs = T_norm[center_row]  # (|C_idx|,)
            for offset, k in enumerate(C_idx.tolist()):
                prob_map[(b, i, int(k))] = float(row_probs[offset].item())

            # ????????????????????????CE??
            pos_cols = []
            for k in C_idx.tolist():
                if (i, int(k)) in pos_lookup.get(int(b), set()):
                    pos_cols.append(int(k))
            if len(pos_cols) > 0:
                target = torch.zeros_like(row_probs)
                # ??????????
                idx_in_row = [int(k) - c_left for k in pos_cols]
                target[torch.tensor(idx_in_row, device=device, dtype=torch.long)] = 1.0 / len(pos_cols)
                ce = -(target * (row_probs.clamp_min(1e-8)).log()).sum()
                loss_terms.append(ce)
                row_count += 1

    loss = torch.stack(loss_terms).mean() if loss_terms else torch.tensor(0.0, device=device)

    # ??? pair_indices ??
    probs_out = torch.zeros(pair_indices.size(0), device=device)
    for idx, (b, e, c) in enumerate(pair_indices.tolist()):
        probs_out[idx] = prob_map.get((int(b), int(e), int(c)), 0.0)

    return loss, probs_out


def train_epoch(model, dataloader, optimizer, scheduler, device, logger, args, pair_criterion):
    """????epoch"""
    model.train()
    total_pair_loss = 0
    total_ot_loss = 0
    total_pair_mix = 0
    total_ot_sup = 0
    total_emotion_loss = 0
    total_cause_loss = 0
    total_loss = 0

    pair_preds = []
    pair_labels_all = []
    emotion_preds = []
    emotion_trues = []
    cause_preds = []
    cause_trues = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Prepare input_ids/attention_mask if available, else create zeros
        if 'input_ids' in batch:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        else:
            # Fallback shapes (B, L)
            B = batch['speakers'].size(0)
            L = batch['speakers'].size(1)
            input_ids = torch.zeros(B, L, dtype=torch.long, device=device)
            attention_mask = torch.zeros(B, L, dtype=torch.long, device=device)
        doc_len = batch['doc_len'].to(device)
        speakers = batch['speakers'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        cause_labels = batch['cause_labels'].to(device)
        pair_indices = batch['pair_indices'].to(device)
        pair_distances = batch['pair_distances'].to(device)
        pair_labels = batch['pair_labels'].to(device)
        texts = batch['texts'] if model.use_doc_encoder else None

        optimizer.zero_grad()

        precomputed_features = batch.get('precomputed_features', None)
        if precomputed_features is not None:
            precomputed_features = precomputed_features.to(device).float()
        outputs = model(
            input_ids, attention_mask,
            doc_len, speakers,
            pair_indices, pair_distances,
            texts=texts,
            precomputed_features=precomputed_features
        )

        pair_logits = outputs['pair_logits']
        emotion_logits = outputs['emotion_logits']
        cause_logits = outputs['cause_logits']

        if getattr(args, 'fgw_only', False):
            # ?FGW?????FGW???CE??????
            fgw_loss, fgw_probs = compute_local_fgw_loss_and_probs(
                model, outputs, batch, device, args, mode='train'
            )
            pair_loss = fgw_loss
            ot_loss = torch.tensor(0.0, device=device)
            ot_sup_loss = torch.tensor(0.0, device=device)
            pair_loss_mix = fgw_loss
            pair_probs_for_metrics = fgw_probs
        else:
            # pair?????????
            pair_loss = pair_criterion(pair_logits, pair_labels)

            ot_pair_scores = outputs.get('ot_pair_scores')
            if getattr(args, 'use_ot_head', False) and ot_pair_scores is not None:
                pair_probs = torch.softmax(pair_logits, dim=-1)[:, 1]

                if args.ot_loss == 'bce':
                    ot_loss = F.binary_cross_entropy(pair_probs, ot_pair_scores)
                elif args.ot_loss == 'mse':
                    ot_loss = F.mse_loss(pair_probs, ot_pair_scores)
                else:  # 'kl'
                    eps = 1e-6
                    p = pair_probs.clamp(eps, 1.0 - eps)
                    q = ot_pair_scores.clamp(eps, 1.0 - eps)
                    ot_loss = (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()

                lam = getattr(args, 'ot_lambda', 0.3)
                pair_loss_mix = (1.0 - lam) * pair_loss + lam * ot_loss

                # OT?????????/???
                ot_sup_weight = getattr(args, 'ot_sup_weight', 1.0)
                if ot_sup_weight > 0:
                    ot_sup_loss = self_supervised_ot_loss(
                        ot_pair_scores, pair_indices, pair_labels,
                        batch['pair_conversation_id'], getattr(args, 'fgw_prob_threshold', 0.0)
                    )
                else:
                    ot_sup_loss = torch.tensor(0.0, device=device)
            else:
                ot_loss = torch.tensor(0.0, device=device)
                ot_sup_loss = torch.tensor(0.0, device=device)
                pair_loss_mix = pair_loss
            pair_probs_for_metrics = torch.softmax(pair_logits, dim=-1)[:, 1]

            # Optional: FGW+MLP fused probabilities for metrics/decoding
            if getattr(args, 'fgw_fuse_row_decode', False) and hasattr(model, 'ot_head') and model.ot_head is not None:
                # FGW scores from OT head (already computed in outputs if use_ot_head)
                fgw_scores = outputs.get('ot_pair_scores')
                if fgw_scores is None:
                    fgw_scores, _ = model.ot_head(
                        outputs['emotion_context'], outputs['cause_context'],
                        outputs['edge_weights'], pair_indices, doc_len,
                        pred_future_cause=getattr(args, 'pred_future_cause', True),
                        max_pair_distance=getattr(args, 'train_max_pair_distance', None)
                    )
                T = max(getattr(args, 'mlp_temp', 1.0), 1e-6)
                mlp_probs_temp = torch.softmax(pair_logits / T, dim=-1)[:, 1]
                lam = getattr(args, 'fgw_blend_lambda', 0.5)
                if getattr(args, 'fgw_blend_space', 'prob') == 'logit':
                    eps = 1e-6
                    fgw_p = torch.clamp(fgw_scores, eps, 1 - eps)
                    mlp_p = torch.clamp(mlp_probs_temp, eps, 1 - eps)
                    fgw_logit = torch.log(fgw_p) - torch.log(1 - fgw_p)
                    mlp_logit = torch.log(mlp_p) - torch.log(1 - mlp_p)
                    logit_blend = lam * fgw_logit + (1.0 - lam) * mlp_logit
                    pair_probs_for_metrics = torch.sigmoid(logit_blend)
                else:
                    pair_probs_for_metrics = lam * fgw_scores + (1.0 - lam) * mlp_probs_temp


        # ???????????/????
        max_len = speakers.size(1)
        doc_mask = MaskGenerator.create_padding_mask(doc_len, max_len).to(device)
        emotion_logits_masked = emotion_logits[doc_mask]
        cause_logits_masked = cause_logits[doc_mask]
        emotion_labels_masked = emotion_labels[doc_mask]
        cause_labels_masked = cause_labels[doc_mask]

        emotion_loss = F.cross_entropy(
            emotion_logits_masked, emotion_labels_masked
        )
        cause_loss = F.cross_entropy(
            cause_logits_masked, cause_labels_masked
        )

        loss = pair_loss_mix + ot_sup_loss * getattr(args, 'ot_sup_weight', 1.0) + args.emotion_weight * emotion_loss + args.cause_weight * cause_loss
        loss.backward()

        if args.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_pair_loss += pair_loss.item()
        total_ot_loss += ot_loss.item() if torch.is_tensor(ot_loss) else float(ot_loss)
        total_pair_mix += pair_loss_mix.item()
        total_ot_sup += ot_sup_loss.item() if torch.is_tensor(ot_sup_loss) else float(ot_sup_loss)
        total_emotion_loss += emotion_loss.item()
        total_cause_loss += cause_loss.item()
        total_loss += loss.item()

        if getattr(args, 'fgw_only', False):
            from collections import defaultdict
            preds = torch.zeros_like(pair_probs_for_metrics, dtype=torch.long)
            key_to_rows = defaultdict(list)
            conv_ids_cpu = batch['pair_conversation_id']
            emo_rows_cpu = pair_indices[:, 1].detach().cpu().tolist()
            for row_idx, (cid, eidx) in enumerate(zip(conv_ids_cpu, emo_rows_cpu)):
                key_to_rows[(cid, eidx)].append(row_idx)
            strategy = getattr(args, 'fgw_pred_strategy', 'topp')
            for _, rows in key_to_rows.items():
                r = torch.tensor(rows, device=pair_probs_for_metrics.device)
                local = pair_probs_for_metrics[r]
                if strategy == 'threshold':
                    thr = getattr(args, 'fgw_threshold', 0.5)
                    mask = local >= thr
                    if not mask.any():
                        top = torch.argmax(local)
                        preds[r[top]] = 1
                    else:
                        preds[r[mask]] = 1
                elif strategy == 'topk':
                    k = int(getattr(args, 'fgw_top_k', 1))
                    k = max(1, min(k, local.numel()))
                    vals, idx = torch.topk(local, k, largest=True)
                    preds[r[idx]] = 1
                elif strategy == 'topp':
                    p = float(getattr(args, 'fgw_top_p', 0.7))
                    p = min(max(p, 0.0), 1.0)
                    vals, idx = torch.sort(local, descending=True)
                    if vals.numel() == 0:
                        continue
                    cum = torch.cumsum(vals, dim=0)
                    pos = (cum >= p).nonzero(as_tuple=False)
                    k_sel = 1 if pos.numel() == 0 else int(pos[0].item()) + 1
                    sel = idx[:k_sel]
                    preds[r[sel]] = 1
                else:
                    top = torch.argmax(local)
                    preds[r[top]] = 1
        elif getattr(args, 'fgw_fuse_row_decode', False):
            from collections import defaultdict
            preds = torch.zeros_like(pair_probs_for_metrics, dtype=torch.long)
            key_to_rows = defaultdict(list)
            conv_ids_cpu = batch['pair_conversation_id']
            emo_rows_cpu = pair_indices[:, 1].detach().cpu().tolist()
            for row_idx, (cid, eidx) in enumerate(zip(conv_ids_cpu, emo_rows_cpu)):
                key_to_rows[(cid, eidx)].append(row_idx)
            strategy = getattr(args, 'fgw_pred_strategy', 'topp')
            for _, rows in key_to_rows.items():
                r = torch.tensor(rows, device=pair_probs_for_metrics.device)
                local = pair_probs_for_metrics[r]
                if strategy == 'threshold':
                    thr = getattr(args, 'fgw_threshold', 0.5)
                    mask = local >= thr
                    if not mask.any():
                        top = torch.argmax(local)
                        preds[r[top]] = 1
                    else:
                        preds[r[mask]] = 1
                elif strategy == 'topk':
                    k = int(getattr(args, 'fgw_top_k', 1))
                    k = max(1, min(k, local.numel()))
                    _, idx = torch.topk(local, k, largest=True)
                    preds[r[idx]] = 1
                elif strategy == 'topp':
                    p = float(getattr(args, 'fgw_top_p', 0.7))
                    p = min(max(p, 0.0), 1.0)
                    vals, idx = torch.sort(local, descending=True)
                    if vals.numel() == 0:
                        continue
                    cum = torch.cumsum(vals, dim=0)
                    pos = (cum >= p).nonzero(as_tuple=False)
                    k_sel = 1 if pos.numel() == 0 else int(pos[0].item()) + 1
                    sel = idx[:k_sel]
                    preds[r[sel]] = 1
                else:
                    top = torch.argmax(local)
                    preds[r[top]] = 1
        else:
            preds = torch.argmax(pair_logits, dim=-1)
        pair_preds.extend(preds.detach().cpu().tolist())
        pair_labels_all.extend(pair_labels.detach().cpu().tolist())

        # ????????????
        emo_pred = torch.argmax(emotion_logits, dim=-1)
        cause_pred = torch.argmax(cause_logits, dim=-1)

        mask_cpu = doc_mask.cpu()
        emotion_preds.extend(emo_pred.cpu()[mask_cpu].tolist())
        emotion_trues.extend(emotion_labels.cpu()[mask_cpu].tolist())
        cause_preds.extend(cause_pred.cpu()[mask_cpu].tolist())
        cause_trues.extend(cause_labels.cpu()[mask_cpu].tolist())

        if batch_idx % args.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.learning_rate
            if getattr(args, 'fgw_only', False):
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'FGW': f'{pair_loss_mix.item():.4f}',
                    'Emo': f'{emotion_loss.item():.4f}',
                    'Cause': f'{cause_loss.item():.4f}',
                    'LR': f'{current_lr:.6f}'
                })
            elif getattr(args, 'use_ot_head', False):
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Pair(mix)': f'{pair_loss_mix.item():.4f}',
                    'Pair(ce)': f'{pair_loss.item():.4f}',
                    'OT': f'{float(ot_loss):.4f}',
                    'OT_sup': f'{float(ot_sup_loss):.4f}',
                    'Emo': f'{emotion_loss.item():.4f}',
                    'Cause': f'{cause_loss.item():.4f}',
                    'LR': f'{current_lr:.6f}'
                })
            else:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Pair': f'{pair_loss.item():.4f}',
                    'Emo': f'{emotion_loss.item():.4f}',
                    'Cause': f'{cause_loss.item():.4f}',
                    'LR': f'{current_lr:.6f}'
                })

    num_batches = len(dataloader)
    avg_pair_loss = total_pair_loss / num_batches
    avg_ot_loss = total_ot_loss / num_batches if getattr(args, 'use_ot_head', False) else 0.0
    avg_pair_mix = total_pair_mix / num_batches if getattr(args, 'use_ot_head', False) else avg_pair_loss
    avg_ot_sup = total_ot_sup / num_batches if getattr(args, 'use_ot_head', False) else 0.0
    avg_emotion_loss = total_emotion_loss / num_batches
    avg_cause_loss = total_cause_loss / num_batches
    avg_total_loss = total_loss / num_batches

    pair_precision, pair_recall, pair_f1 = MetricsCalculator.calculate_prf(
        torch.tensor(pair_preds), torch.tensor(pair_labels_all)
    )

    # ???????
    if len(emotion_preds) > 0:
        emotion_preds_tensor = torch.tensor(emotion_preds)
        emotion_trues_tensor = torch.tensor(emotion_trues)
        cause_preds_tensor = torch.tensor(cause_preds)
        cause_trues_tensor = torch.tensor(cause_trues)
        emotion_p, emotion_r, emotion_f1 = MetricsCalculator.calculate_prf(
            emotion_preds_tensor, emotion_trues_tensor,
            average='macro' if args.use_emocate else 'binary',
            use_emocate=args.use_emocate
        )
        cause_p, cause_r, cause_f1 = MetricsCalculator.calculate_prf(
            cause_preds_tensor, cause_trues_tensor
        )
    else:
        emotion_p = emotion_r = emotion_f1 = 0.0
        cause_p = cause_r = cause_f1 = 0.0

    if getattr(args, 'fgw_only', False):
        logger.info(
            "Train - Loss(total/fgw/emotion/cause): {:.4f}/{:.4f}/{:.4f}/{:.4f}, Pair P/R/F1: {:.4f}/{:.4f}/{:.4f}, Emotion P/R/F1: {:.4f}/{:.4f}/{:.4f}, Cause P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
                avg_total_loss, avg_pair_mix, avg_emotion_loss, avg_cause_loss,
                pair_precision, pair_recall, pair_f1,
                emotion_p, emotion_r, emotion_f1,
                cause_p, cause_r, cause_f1
            )
        )
    elif getattr(args, 'use_ot_head', False):
        logger.info(
            "Train - Loss(total/pair_mix/ot/ot_sup/pair_ce/emotion/cause): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}, Pair P/R/F1: {:.4f}/{:.4f}/{:.4f}, Emotion P/R/F1: {:.4f}/{:.4f}/{:.4f}, Cause P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
                avg_total_loss, avg_pair_mix, avg_ot_loss, avg_ot_sup, avg_pair_loss, avg_emotion_loss, avg_cause_loss,
                pair_precision, pair_recall, pair_f1,
                emotion_p, emotion_r, emotion_f1,
                cause_p, cause_r, cause_f1
            )
        )
    else:
        logger.info(
            "Train - Loss(total/pair/emotion/cause): {:.4f}/{:.4f}/{:.4f}/{:.4f}, Pair P/R/F1: {:.4f}/{:.4f}/{:.4f}, Emotion P/R/F1: {:.4f}/{:.4f}/{:.4f}, Cause P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
                avg_total_loss, avg_pair_loss, avg_emotion_loss, avg_cause_loss,
                pair_precision, pair_recall, pair_f1,
                emotion_p, emotion_r, emotion_f1,
                cause_p, cause_r, cause_f1
            )
        )

    return {
        'loss': avg_total_loss,
        'pair_loss': avg_pair_loss,
        'emotion_loss': avg_emotion_loss,
        'cause_loss': avg_cause_loss,
        'pair_metrics': (pair_precision, pair_recall, pair_f1),
        'emotion_metrics': (emotion_p, emotion_r, emotion_f1),
        'cause_metrics': (cause_p, cause_r, cause_f1)
    }


def evaluate(model, dataloader, device, logger, args, threshold=None, pair_criterion=None):
    """????"""
    model.eval()
    total_pair_loss = 0
    total_emotion_loss = 0
    total_cause_loss = 0
    total_loss = 0

    pair_probs_all = []
    pair_labels_all = []
    pair_preds_all = []
    convo_ids_all = []
    emo_ids_all = []
    cause_ids_all = []

    emotion_preds_all = []
    emotion_trues_all = []
    cause_preds_all = []
    cause_trues_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # ???????????? input_ids/attention_mask
            if 'input_ids' in batch:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            else:
                B = batch['speakers'].size(0)
                L = batch['speakers'].size(1)
                input_ids = torch.zeros(B, L, dtype=torch.long, device=device)
                attention_mask = torch.zeros(B, L, dtype=torch.long, device=device)
            doc_len = batch['doc_len'].to(device)
            speakers = batch['speakers'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            cause_labels = batch['cause_labels'].to(device)
            pair_indices = batch['pair_indices'].to(device)
            pair_distances = batch['pair_distances'].to(device)
            pair_labels = batch['pair_labels'].to(device)
            texts = batch['texts'] if model.use_doc_encoder else None

            precomputed_features = batch.get('precomputed_features', None)
            if precomputed_features is not None:
                precomputed_features = precomputed_features.to(device).float()
            outputs = model(
                input_ids, attention_mask,
                doc_len, speakers,
                pair_indices, pair_distances,
                texts=texts,
                precomputed_features=precomputed_features
            )

            pair_logits = outputs['pair_logits']
            emotion_logits = outputs['emotion_logits']
            cause_logits = outputs['cause_logits']

            # ??????
            doc_mask = MaskGenerator.create_padding_mask(doc_len, input_ids.size(1)).to(device)
            emotion_logits_masked = emotion_logits[doc_mask]
            cause_logits_masked = cause_logits[doc_mask]
            emotion_labels_masked = emotion_labels[doc_mask]
            cause_labels_masked = cause_labels[doc_mask]

            emotion_loss = F.cross_entropy(
                emotion_logits_masked, emotion_labels_masked
            )
            cause_loss = F.cross_entropy(
                cause_logits_masked, cause_labels_masked
            )

            if getattr(args, 'fgw_only', False):
                fgw_loss, probs = compute_local_fgw_loss_and_probs(
                    model, outputs, batch, device, args, mode='eval'
                )
                pair_loss = fgw_loss
                total_pair_loss += pair_loss.item()
                total_emotion_loss += emotion_loss.item()
                total_cause_loss += cause_loss.item()
                loss = pair_loss + args.emotion_weight * emotion_loss + args.cause_weight * cause_loss
                total_loss += loss.item()
                # ??
                # FGW-only: ?????????top-p?
                from collections import defaultdict
                preds = torch.zeros_like(probs, dtype=torch.long)
                key_to_rows = defaultdict(list)
                conv_ids_cpu = batch['pair_conversation_id']
                emo_rows_cpu = pair_indices[:, 1].cpu().tolist()
                for row_idx, (cid, eidx) in enumerate(zip(conv_ids_cpu, emo_rows_cpu)):
                    key_to_rows[(cid, eidx)].append(row_idx)
                strategy = getattr(args, 'fgw_pred_strategy', 'topp')
                for _, rows in key_to_rows.items():
                    r = torch.tensor(rows, device=probs.device)
                    local = probs[r]
                    if strategy == 'threshold':
                        thr = threshold if threshold is not None else getattr(args, 'fgw_threshold', 0.5)
                        mask = local >= thr
                        if not mask.any():
                            # ?????top-1
                            top = torch.argmax(local)
                            preds[r[top]] = 1
                        else:
                            preds[r[mask]] = 1
                    elif strategy == 'topk':
                        k = int(getattr(args, 'fgw_top_k', 1))
                        k = max(1, min(k, local.numel()))
                        vals, idx = torch.topk(local, k, largest=True)
                        preds[r[idx]] = 1
                    elif strategy == 'topp':
                        p = float(getattr(args, 'fgw_top_p', 0.7))
                        p = min(max(p, 0.0), 1.0)
                        vals, idx = torch.sort(local, descending=True)
                        if vals.numel() == 0:
                            continue
                        cum = torch.cumsum(vals, dim=0)
                        pos = (cum >= p).nonzero(as_tuple=False)
                        k_sel = 1 if pos.numel() == 0 else int(pos[0].item()) + 1
                        sel = idx[:k_sel]
                        preds[r[sel]] = 1
                    else:  # argmax
                        top = torch.argmax(local)
                        preds[r[top]] = 1
            else:
                # pair?????????
                if pair_criterion is None:
                    pair_loss = F.cross_entropy(pair_logits, pair_labels)
                else:
                    pair_loss = pair_criterion(pair_logits, pair_labels)

                loss = pair_loss + args.emotion_weight * emotion_loss + args.cause_weight * cause_loss
                total_pair_loss += pair_loss.item()
                total_emotion_loss += emotion_loss.item()
                total_cause_loss += cause_loss.item()
                total_loss += loss.item()

                T_eval = max(getattr(args, 'mlp_temp', 1.0), 1e-6) if getattr(args, 'fgw_fuse_row_decode', False) else 1.0
                probs = torch.softmax(pair_logits / T_eval, dim=-1)[:, 1]

                # 统一的评估期融合：若开启行内融合且存在OT头，则使用一次FGW分数并融合
                if getattr(args, 'fgw_fuse_row_decode', False) and hasattr(model, 'ot_head') and model.ot_head is not None:
                    fgw_scores = outputs.get('ot_pair_scores')
                    if fgw_scores is None:
                        fgw_scores, _ = model.ot_head(
                            outputs['emotion_context'], outputs['cause_context'],
                            outputs['edge_weights'], pair_indices, doc_len,
                            pred_future_cause=getattr(args, 'pred_future_cause', True),
                            max_pair_distance=getattr(args, 'eval_max_pair_distance', None)
                        )
                    if args.fgw_mode == 'replace':
                        probs = fgw_scores
                    else:
                        lam = getattr(args, 'fgw_blend_lambda', 0.5)
                        if getattr(args, 'fgw_blend_space', 'prob') == 'logit':
                            eps = 1e-6
                            p1 = torch.clamp(fgw_scores, eps, 1 - eps)
                            p2 = torch.clamp(probs, eps, 1 - eps)
                            logit1 = torch.log(p1) - torch.log(1 - p1)
                            logit2 = torch.log(p2) - torch.log(1 - p2)
                            probs = torch.sigmoid(lam * logit1 + (1.0 - lam) * logit2)
                        else:
                            probs = lam * fgw_scores + (1.0 - lam) * probs

                if getattr(args, 'fgw_fuse_row_decode', False):
                    # Row-wise decoding per (conversation, emotion) using fused probs
                    from collections import defaultdict
                    preds = torch.zeros_like(probs, dtype=torch.long)
                    key_to_rows = defaultdict(list)
                    conv_ids_cpu = batch['pair_conversation_id']
                    emo_rows_cpu = pair_indices[:, 1].detach().cpu().tolist()
                    for row_idx, (cid, eidx) in enumerate(zip(conv_ids_cpu, emo_rows_cpu)):
                        key_to_rows[(cid, eidx)].append(row_idx)
                    strategy = getattr(args, 'fgw_pred_strategy', 'topp')
                    for _, rows in key_to_rows.items():
                        r = torch.tensor(rows, device=probs.device)
                        local = probs[r]
                        if strategy == 'threshold':
                            thr = getattr(args, 'fgw_threshold', 0.5)
                            mask = local >= thr
                            if not mask.any():
                                top = torch.argmax(local)
                                preds[r[top]] = 1
                            else:
                                preds[r[mask]] = 1
                        elif strategy == 'topk':
                            k = int(getattr(args, 'fgw_top_k', 1))
                            k = max(1, min(k, local.numel()))
                            vals, idx = torch.topk(local, k, largest=True)
                            preds[r[idx]] = 1
                        elif strategy == 'topp':
                            p = float(getattr(args, 'fgw_top_p', 0.7))
                            p = min(max(p, 0.0), 1.0)
                            vals, idx = torch.sort(local, descending=True)
                            if vals.numel() == 0:
                                continue
                            cum = torch.cumsum(vals, dim=0)
                            pos = (cum >= p).nonzero(as_tuple=False)
                            k_sel = 1 if pos.numel() == 0 else int(pos[0].item()) + 1
                            sel = idx[:k_sel]
                            preds[r[sel]] = 1
                        else:
                            top = torch.argmax(local)
                            preds[r[top]] = 1
                else:
                    if threshold is not None:
                        preds = (probs >= threshold).long()
                    else:
                        preds = torch.argmax(pair_logits, dim=-1)

            pair_probs_all.extend(probs.cpu().tolist())
            pair_labels_all.extend(pair_labels.cpu().tolist())
            pair_preds_all.extend(preds.cpu().tolist())

            convo_ids_all.extend(batch['pair_conversation_id'])
            emo_ids_all.extend(pair_indices[:, 1].cpu().tolist())
            cause_ids_all.extend(pair_indices[:, 2].cpu().tolist())

            # ?????????????
            emo_pred = torch.argmax(emotion_logits, dim=-1)
            cause_pred = torch.argmax(cause_logits, dim=-1)
            mask_cpu = doc_mask.cpu()
            emotion_preds_all.extend(emo_pred.cpu()[mask_cpu].tolist())
            emotion_trues_all.extend(emotion_labels.cpu()[mask_cpu].tolist())
            cause_preds_all.extend(cause_pred.cpu()[mask_cpu].tolist())
            cause_trues_all.extend(cause_labels.cpu()[mask_cpu].tolist())

    num_batches = len(dataloader)
    avg_pair_loss = total_pair_loss / num_batches
    avg_emotion_loss = total_emotion_loss / num_batches
    avg_cause_loss = total_cause_loss / num_batches
    avg_total_loss = total_loss / num_batches

    pair_precision, pair_recall, pair_f1 = MetricsCalculator.calculate_prf(
        torch.tensor(pair_preds_all), torch.tensor(pair_labels_all)
    )

    if len(emotion_preds_all) > 0:
        emotion_p, emotion_r, emotion_f1 = MetricsCalculator.calculate_prf(
            torch.tensor(emotion_preds_all), torch.tensor(emotion_trues_all),
            average='macro' if args.use_emocate else 'binary',
            use_emocate=args.use_emocate
        )
        cause_p, cause_r, cause_f1 = MetricsCalculator.calculate_prf(
            torch.tensor(cause_preds_all), torch.tensor(cause_trues_all)
        )
    else:
        emotion_p = emotion_r = emotion_f1 = 0.0
        cause_p = cause_r = cause_f1 = 0.0

    pred_pairs_by_conv = defaultdict(list)
    true_pairs_by_conv_candidates = defaultdict(list)

    for conv_id, emo_idx, cause_idx, pred_label, true_label in zip(
        convo_ids_all, emo_ids_all, cause_ids_all, pair_preds_all, pair_labels_all
    ):
        if true_label == 1:
            true_pairs_by_conv_candidates[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))
        if pred_label == 1:
            pred_pairs_by_conv[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))

    true_pairs_by_conv_full = getattr(dataloader.dataset, 'true_pairs_by_conv_full', {})
    all_conv_ids_candidates = set(true_pairs_by_conv_candidates.keys()) | set(pred_pairs_by_conv.keys())
    all_true_pairs_candidates = [true_pairs_by_conv_candidates.get(conv_id, []) for conv_id in all_conv_ids_candidates]
    all_pred_pairs_candidates = [pred_pairs_by_conv.get(conv_id, []) for conv_id in all_conv_ids_candidates]

    if all_conv_ids_candidates:
        pair_metrics_candidates = PairEvaluator.evaluate_pairs(all_true_pairs_candidates, all_pred_pairs_candidates)
    else:
        pair_metrics_candidates = {'precision': 0, 'recall': 0, 'f1': 0}

    all_conv_ids_full = set(true_pairs_by_conv_full.keys()) | set(pred_pairs_by_conv.keys())
    all_true_pairs_full = [true_pairs_by_conv_full.get(conv_id, []) for conv_id in all_conv_ids_full]
    all_pred_pairs_full = [pred_pairs_by_conv.get(conv_id, []) for conv_id in all_conv_ids_full]

    if all_conv_ids_full:
        pair_metrics_full = PairEvaluator.evaluate_pairs(all_true_pairs_full, all_pred_pairs_full)
    else:
        pair_metrics_full = {'precision': 0, 'recall': 0, 'f1': 0}

    if getattr(args, 'fgw_only', False):
        logger.info(
            "Eval - Loss(total/fgw/emotion/cause): {:.4f}/{:.4f}/{:.4f}/{:.4f}, Pair P/R/F1: {:.4f}/{:.4f}/{:.4f}, Emotion P/R/F1: {:.4f}/{:.4f}/{:.4f}, Cause P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
                avg_total_loss, avg_pair_loss, avg_emotion_loss, avg_cause_loss,
                pair_precision, pair_recall, pair_f1,
                emotion_p, emotion_r, emotion_f1,
                cause_p, cause_r, cause_f1
            )
        )
    else:
        logger.info(
            "Eval - Loss(total/pair/emotion/cause): {:.4f}/{:.4f}/{:.4f}/{:.4f}, Pair P/R/F1: {:.4f}/{:.4f}/{:.4f}, Emotion P/R/F1: {:.4f}/{:.4f}/{:.4f}, Cause P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
                avg_total_loss, avg_pair_loss, avg_emotion_loss, avg_cause_loss,
                pair_precision, pair_recall, pair_f1,
                emotion_p, emotion_r, emotion_f1,
                cause_p, cause_r, cause_f1
            )
        )
    logger.info(
        "候选口径 P/R/F1: {:.4f}/{:.4f}/{:.4f}, 全口径 P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
            pair_metrics_candidates['precision'], pair_metrics_candidates['recall'], pair_metrics_candidates['f1'],
            pair_metrics_full['precision'], pair_metrics_full['recall'], pair_metrics_full['f1']
        )
    )

    pair_precision_full = pair_metrics_full.get('precision', 0.0)
    pair_recall_full = pair_metrics_full.get('recall', 0.0)
    pair_f1_full = pair_metrics_full.get('f1', 0.0)
    pair_precision_candidates = pair_metrics_candidates.get('precision', 0.0)
    pair_recall_candidates = pair_metrics_candidates.get('recall', 0.0)
    pair_f1_candidates = pair_metrics_candidates.get('f1', 0.0)

    return {
        'loss': avg_total_loss,
        'pair_loss': avg_pair_loss,
        'emotion_loss': avg_emotion_loss,
        'cause_loss': avg_cause_loss,
        'pair_metrics': (pair_precision, pair_recall, pair_f1),
        'emotion_metrics': (emotion_p, emotion_r, emotion_f1),
        'cause_metrics': (cause_p, cause_r, cause_f1),
        'pair_metrics_candidates': pair_metrics_candidates,
        'pair_metrics_full': pair_metrics_full,
        'pair_precision_candidates': pair_precision_candidates,
        'pair_recall_candidates': pair_recall_candidates,
        'pair_f1_candidates': pair_f1_candidates,
        'pair_precision_full': pair_precision_full,
        'pair_recall_full': pair_recall_full,
        'pair_f1_full': pair_f1_full,
        'pair_probs': pair_probs_all,
        'pair_labels': pair_labels_all,
        'pair_preds': pair_preds_all,
        'convo_ids': convo_ids_all,
        'emo_ids': emo_ids_all,
        'cause_ids': cause_ids_all
    }


def main():
    # ????
    config = Config()
    args = config.parse_args()

    # FGW-only mode: ensure OT head is enabled
    if getattr(args, 'fgw_only', False):
        args.use_ot_head = True

    # ??????????????step???
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ??????
    set_seed(args.seed)

    # ????
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ????
    logger = setup_logging(args.log_dir, args.dataset, args.model_type)

    print_time()
    logger.info("开始运行...")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"设备: {device}")

    # ???????
    logger.info("加载数据...")
    train_loader, test_loader, dev_loader, _ = create_feature_data_loaders(
        dataset_name=args.dataset,
        feature_dir=getattr(args, 'feature_dir', './features'),
        batch_size=args.batch_size,
        pred_future_cause=args.pred_future_cause,
        use_emocate=args.use_emocate,
        use_emotion_category=args.use_emotion_category,
        negative_sampling_ratio=getattr(args, 'negative_sampling_ratio', 3.0),
        eval_max_pair_distance=getattr(args, 'eval_max_pair_distance', None),
        train_max_pair_distance=getattr(args, 'train_max_pair_distance', None),
        max_doc_len=args.max_doc_len
    )

    # ???????????

    logger.info(f"训练集: {len(train_loader)} 批")
    logger.info(f"测试集: {len(test_loader)} 批")
    if dev_loader:
        logger.info(f"验证集: {len(dev_loader)} 批")

    # ????
    logger.info("构建模型...")
    model_config = get_model_config(args)
    model = MECPE_Model(model_config).to(device)

    # ??????
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数总数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    # ?????
    if args.model_type == 'bert':
        # BERT??AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8
        )

        # ??????
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    else:
        # BiLSTM??Adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler = None

    # ??????
    logger.info("计算类别权重...")
    weight_ratio_cap = getattr(args, 'weight_ratio_cap', 4.0)  # ?????????4.0
    class_weights = PairLoss.compute_class_weights(train_loader, device, weight_ratio_cap)
    criterion = PairLoss(class_weight=class_weights)
    logger.info(f"Pair CrossEntropyLoss 类别权重: {class_weights}")

    # ???????
    early_stopping = EarlyStopping(patience=args.patience)
    model_saver = ModelSaver(args.save_dir)

    # ??????????
    start_epoch = 0
    best_f1 = 0
    if args.load_checkpoint:
        start_epoch, metrics = model_saver.load_checkpoint(model, optimizer, args.load_checkpoint)
        if metrics:
            best_f1 = metrics.get('best_f1', 0)
            logger.info(f"从检查点恢复: 起始epoch={start_epoch}, 最佳F1={best_f1}")

    # ????
    logger.info("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # ??
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, logger, args, pair_criterion=criterion)

        # ??
        if (epoch + 1) % args.eval_interval == 0:
            if dev_loader:
                eval_metrics = evaluate(model, dev_loader, device, logger, args, threshold=None, pair_criterion=criterion)
                eval_f1 = eval_metrics['pair_f1_full']  # ?????F1??????
            else:
                eval_metrics = evaluate(model, test_loader, device, logger, args, threshold=None, pair_criterion=criterion)
                eval_f1 = eval_metrics['pair_f1_full']  # ?????F1??????

            # ??????
            is_best = eval_f1 > best_f1
            if is_best:
                best_f1 = eval_f1
                logger.info(f"当前最佳F1: {best_f1:.4f}")

            # ?????
            if (epoch + 1) % args.save_interval == 0 or is_best:
                metrics = {
                    'epoch': epoch + 1,
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'best_f1': best_f1
                }
                model_saver.save_checkpoint(model, optimizer, epoch + 1, metrics, is_best)

            # ????
            if early_stopping(eval_metrics['loss'], model):
                logger.info("早停触发")
                break

    # ????
    logger.info("\n=== 训练结束 ===")
    model_saver.load_checkpoint(model, filename='best_model.pt')

    # ????????????
    optimal_threshold = None
    if dev_loader:
        logger.info("\n=== 在验证集上搜索阈值 ===")
        optimal_threshold = find_optimal_threshold(model, dev_loader, device, logger, args)
        logger.info(f"最佳阈值: {optimal_threshold:.3f}")

    # ???????????
    logger.info("\n--- 测试（阈值0.5）评估 ---")
    test_metrics_default = evaluate(model, test_loader, device, logger, args, threshold=None, pair_criterion=criterion)
    emo_p, emo_r, emo_f1 = test_metrics_default['emotion_metrics']
    cau_p, cau_r, cau_f1 = test_metrics_default['cause_metrics']
    logger.info(f"  情感 - P: {emo_p:.4f}, R: {emo_r:.4f}, F1: {emo_f1:.4f}")
    logger.info(f"  原因 - P: {cau_p:.4f}, R: {cau_r:.4f}, F1: {cau_f1:.4f}")
    logger.info(f"  对（候选口径） - P: {test_metrics_default['pair_precision_candidates']:.4f}, "
                f"R: {test_metrics_default['pair_recall_candidates']:.4f}, "
                f"F1: {test_metrics_default['pair_f1_candidates']:.4f}")
    logger.info(f"  对（全口径） - P: {test_metrics_default['pair_precision_full']:.4f}, "
                f"R: {test_metrics_default['pair_recall_full']:.4f}, "
                f"F1: {test_metrics_default['pair_f1_full']:.4f}")

    # ????????????????
    if optimal_threshold is not None:
        logger.info(f"\n--- 测试（最优阈值 {optimal_threshold:.3f}）评估 ---")
        test_metrics_calibrated = evaluate(model, test_loader, device, logger, args, threshold=optimal_threshold, pair_criterion=criterion)
        emo_p2, emo_r2, emo_f12 = test_metrics_calibrated['emotion_metrics']
        cau_p2, cau_r2, cau_f12 = test_metrics_calibrated['cause_metrics']
        logger.info(f"  情感 - P: {emo_p2:.4f}, R: {emo_r2:.4f}, F1: {emo_f12:.4f}")
        logger.info(f"  原因 - P: {cau_p2:.4f}, R: {cau_r2:.4f}, F1: {cau_f12:.4f}")
        logger.info(f"  对（候选口径） - P: {test_metrics_calibrated['pair_precision_candidates']:.4f}, "
                    f"R: {test_metrics_calibrated['pair_recall_candidates']:.4f}, "
                    f"F1: {test_metrics_calibrated['pair_f1_candidates']:.4f}")
        logger.info(f"  对（全口径） - P: {test_metrics_calibrated['pair_precision_full']:.4f}, "
                    f"R: {test_metrics_calibrated['pair_recall_full']:.4f}, "
                    f"F1: {test_metrics_calibrated['pair_f1_full']:.4f}")

        # ?????????????
        f1_improvement = test_metrics_calibrated['pair_f1_full'] - test_metrics_default['pair_f1_full']
        p_improvement = test_metrics_calibrated['pair_precision_full'] - test_metrics_default['pair_precision_full']
        r_change = test_metrics_calibrated['pair_recall_full'] - test_metrics_default['pair_recall_full']
        logger.info(f"\n--- 对比（默认 vs 最优阈值） ---")
        logger.info(f"  精确率变化: {p_improvement:+.4f}")
        logger.info(f"  召回率变化: {r_change:+.4f}")
        logger.info(f"  F1变化: {f1_improvement:+.4f}")

    print_time()
    logger.info("全部完成")


if __name__ == "__main__":
    main()

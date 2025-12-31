# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
from tqdm import tqdm
from collections import defaultdict


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_model_config
from data_loader import create_feature_data_loaders
from models import Model as ECPEC_Model, MaskGenerator
from loss import PairLoss
from decoders import global_mcmf_decode
from utils import (
    set_seed, print_time, setup_logging, 
    MetricsCalculator, PairEvaluator, EarlyStopping, ModelSaver
)


def decode_transport_topk(outputs, pair_indices, batch_conv_ids, row_capacity, fgw_row_temp, device):
    """T-based row top-k soft capacity decoding.
    """
    if outputs.get('ot_transports') is None or not outputs['ot_transports']:
        return None
    
    N = pair_indices.size(0)
    preds_topk = torch.zeros(N, dtype=torch.long, device=device)
    
    # Build conversation to row index mapping
    conv_to_rows = defaultdict(list)
    for ridx, cid in enumerate(batch_conv_ids):
        conv_to_rows[cid].append(ridx)
    
    row_k = int(row_capacity)
    for cid, rows in conv_to_rows.items():
        T = outputs['ot_transports'].get(int(cid), None)
        if T is None or T.numel() == 0:
            continue
        
        sub = pair_indices[torch.tensor(rows, dtype=torch.long, device=device)]
        uniq_e_t = torch.unique(sub[:, 1], sorted=True)
        uniq_c_t = torch.unique(sub[:, 2], sorted=True)
        uniq_e = [int(x.item()) for x in uniq_e_t]
        uniq_c = [int(x.item()) for x in uniq_c_t]
        
        Tn = F.softmax(T / max(fgw_row_temp, 1e-6), dim=1)
        

        n_e = min(Tn.size(0), len(uniq_e))
        n_c = min(Tn.size(1), len(uniq_c))
        if n_e == 0 or n_c == 0:
            continue
        
        k = int(min(row_k, n_c))
        _, topk_idx = torch.topk(Tn[:n_e, :n_c], k=k, dim=1)
        
        select_set = set()
        for r in range(n_e):
            e_global = uniq_e[r]
            for j in topk_idx[r].tolist():
                if j < len(uniq_c):
                    c_global = uniq_c[j]
                    select_set.add((e_global, c_global))
        
        for ridx in rows:
            e0 = int(pair_indices[ridx, 1].item())
            c0 = int(pair_indices[ridx, 2].item())
            if (e0, c0) in select_set:
                preds_topk[ridx] = 1
    
    return preds_topk


def train_epoch(model, dataloader, optimizer, device, logger, args, pair_criterion):
    model.train()
    total_pair_loss = 0.0
    total_ot_loss = 0.0
    total_pair_mix = 0.0
    total_emotion_loss = 0.0
    total_cause_loss = 0.0
    total_loss = 0.0
    total_transport_ce = 0.0
    total_distill = 0.0

    pair_preds = []
    pair_labels_all = []
    emotion_preds = []
    emotion_trues = []
    cause_preds = []
    cause_trues = []

    pair_probs_all = []
    pair_preds_all = []
    pair_preds_all_topk = []
    convo_ids_all = []
    emo_ids_all = []
    cause_ids_all = []

    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        doc_len = batch['doc_len'].to(device)
        speakers = batch['speakers'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        cause_labels = batch['cause_labels'].to(device)
        pair_indices = batch['pair_indices'].to(device)
        pair_distances = batch['pair_distances'].to(device)
        pair_labels = batch['pair_labels'].to(device)

        optimizer.zero_grad()

        precomputed_features = batch.get('precomputed_features', None)
        if precomputed_features is not None:
            precomputed_features = precomputed_features.to(device).float()

        outputs = model(
            precomputed_features,
            doc_len, speakers,
            pair_indices, pair_distances
        )

        pair_logits = outputs['pair_logits']
        emotion_logits = outputs['emotion_logits']
        cause_logits = outputs['cause_logits']

        # Pair loss + optional OT distillation
        pair_loss = pair_criterion(pair_logits, pair_labels)
        ot_pair_scores = outputs.get('ot_pair_scores')
        ce_total = pair_loss
        if getattr(args, 'use_ot_head', False) and ot_pair_scores is not None:
            T = max(float(getattr(args, 'mlp_temp', 1.0)), 1e-6)
            z_mlp = (pair_logits[:, 1] - pair_logits[:, 0]) / T
            eps = 1e-6
            p_fgw = ot_pair_scores.clamp(eps, 1 - eps)
            z_fgw = torch.log(p_fgw) - torch.log(1 - p_fgw)
            lam_blend = float(getattr(args, 'fgw_blend_lambda', 0.5))
            z_fuse = (1.0 - lam_blend) * z_mlp + lam_blend * z_fgw
            logits_fuse_2d = torch.stack([-z_fuse, z_fuse], dim=-1)
            ce_total = pair_criterion(logits_fuse_2d, pair_labels)
        if getattr(args, 'use_ot_head', False) and ot_pair_scores is not None:
            pair_probs = torch.softmax(pair_logits, dim=-1)[:, 1]
            if args.ot_loss == 'bce':
                ot_loss = F.binary_cross_entropy(pair_probs, ot_pair_scores)
            elif args.ot_loss == 'mse':
                ot_loss = F.mse_loss(pair_probs, ot_pair_scores)
            else:
                eps = 1e-6
                p = pair_probs.clamp(eps, 1.0 - eps)
                q = ot_pair_scores.clamp(eps, 1.0 - eps)
                ot_loss = (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()
            lam = getattr(args, 'ot_lambda', 0.3)
            pair_loss_mix = (1.0 - lam) * ce_total + lam * ot_loss
        else:
            ot_loss = torch.tensor(0.0, device=device)

            pair_loss_mix = ce_total

        # Generate edge weights for MCMF 
        T = max(getattr(args, 'mlp_temp', 1.0), 1e-6)
        pair_probs_for_metrics = torch.softmax(pair_logits / T, dim=-1)[:, 1]
        if getattr(args, 'use_ot_head', False) and hasattr(model, 'ot_head') and model.ot_head is not None:
            fgw_scores = outputs.get('ot_pair_scores')
            if fgw_scores is None:
                fgw_scores, _ = model.ot_head(
                    outputs['emotion_context'], outputs['cause_context'],
                        outputs['edge_weights_e'], outputs['edge_weights_c'],
                        pair_indices, doc_len,
                    pred_future_cause=getattr(args, 'pred_future_cause', False),
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
                pair_probs_for_metrics = torch.sigmoid(lam * fgw_logit + (1.0 - lam) * mlp_logit)
            else:
                pair_probs_for_metrics = lam * fgw_scores + (1.0 - lam) * mlp_probs_temp

        # Utterance-level losses
        max_len = speakers.size(1)
        doc_mask = MaskGenerator.create_padding_mask(doc_len, max_len).to(device)
        emotion_loss = F.cross_entropy(emotion_logits[doc_mask], emotion_labels[doc_mask])
        cause_loss = F.cross_entropy(cause_logits[doc_mask], cause_labels[doc_mask])


        # Additional: T supervision (BCE), consistency distillation with T (KL)
        transport_ce_loss = torch.tensor(0.0, device=device)
        distill_with_t_loss = torch.tensor(0.0, device=device)
        if getattr(args, 'use_ot_head', False) and outputs.get('ot_pair_scores') is not None:
            t_scores = outputs['ot_pair_scores']
            posw = float(getattr(args, 'transport_ce_pos_weight', 1.0))
            weights = torch.where(pair_labels > 0, torch.tensor(posw, device=device), torch.tensor(1.0, device=device))
            bce = F.binary_cross_entropy(torch.clamp(t_scores, 1e-6, 1-1e-6), pair_labels.float(), reduction='none')
            transport_ce_loss = (bce * weights).mean()

            temp = float(getattr(args, 'distill_with_t_temp', 2.0))
            z = (pair_logits[:, 1] - pair_logits[:, 0]) / max(temp, 1e-6)
            p = torch.sigmoid(z).clamp(1e-6, 1-1e-6)
            q = torch.clamp(t_scores, 1e-6, 1-1e-6)
            def _kl(pv, qv):
                return (pv * torch.log(pv/qv) + (1-pv) * torch.log((1-pv)/(1-qv))).mean()
            ddir = getattr(args, 'distill_with_t_dir', 'mlp_to_t')
            if ddir == 't_to_mlp':
                distill_with_t_loss = _kl(q, p)
            elif ddir == 'sym':
                distill_with_t_loss = 0.5 * (_kl(p, q) + _kl(q, p))
            else:
                distill_with_t_loss = _kl(p, q)

        loss = pair_loss_mix \
               + args.emotion_weight * emotion_loss \
               + args.cause_weight * cause_loss \
               + float(getattr(args, 'transport_ce_lambda', 0.0)) * transport_ce_loss \
               + float(getattr(args, 'distill_with_t_lambda', 0.0)) * distill_with_t_loss 
        loss.backward()
        if args.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        # Accumulate losses
        total_pair_loss += pair_loss.item()
        total_ot_loss += float(ot_loss)
        total_pair_mix += pair_loss_mix.item()
        total_emotion_loss += emotion_loss.item()
        total_cause_loss += cause_loss.item()
        total_loss += loss.item()
        if getattr(args, 'use_ot_head', False) and outputs.get('ot_pair_scores') is not None:
            total_transport_ce += float(transport_ce_loss)
            total_distill += float(distill_with_t_loss)

        # Global MCMF decoding
        preds = global_mcmf_decode(
            pair_indices=pair_indices,
            probs=pair_probs_for_metrics,
            conv_ids=batch['pair_conversation_id'],
            row_cap=getattr(args, 'mcmf_row_capacity', 1),
            col_cap=getattr(args, 'mcmf_col_capacity', 1),
            lambda_cost=getattr(args, 'mcmf_lambda', 0.5),
            score_space=getattr(args, 'mcmf_score_space', 'prob'),
            eps=getattr(args, 'mcmf_eps', 1e-6),
            pre_topk_per_row=getattr(args, 'mcmf_pre_topk_per_row', None),
            pre_min_prob=getattr(args, 'mcmf_pre_min_prob', None),
            pre_min_logit=getattr(args, 'mcmf_pre_min_logit', None),
            device=device,
        )

        pair_preds.extend(preds.detach().cpu().tolist())
        pair_labels_all.extend(pair_labels.detach().cpu().tolist())
        pair_probs_all.extend(pair_probs_for_metrics.detach().cpu().tolist())
        pair_preds_all.extend(preds.detach().cpu().tolist())
        convo_ids_all.extend(batch['pair_conversation_id'])
        emo_ids_all.extend(pair_indices[:, 1].detach().cpu().tolist())
        cause_ids_all.extend(pair_indices[:, 2].detach().cpu().tolist())

        # T-based row top-k soft capacity decoding 
        if getattr(args, 'use_ot_head', False):
            preds_topk = decode_transport_topk(
                outputs, pair_indices, batch['pair_conversation_id'],
                getattr(args, 'mcmf_row_capacity', 1),
                getattr(args, 'fgw_row_temp', 0.7),
                device
            )
            if preds_topk is not None:
                pair_preds_all_topk.extend(preds_topk.detach().cpu().tolist())

        # Utterance-level predictions
        emo_pred = torch.argmax(emotion_logits, dim=-1)
        cause_pred = torch.argmax(cause_logits, dim=-1)
        mask_cpu = doc_mask.cpu()
        emotion_preds.extend(emo_pred.cpu()[mask_cpu].tolist())
        emotion_trues.extend(emotion_labels.cpu()[mask_cpu].tolist())
        cause_preds.extend(cause_pred.cpu()[mask_cpu].tolist())
        cause_trues.extend(cause_labels.cpu()[mask_cpu].tolist())

        if batch_idx % args.log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pf = {
                'Loss': f'{loss.item():.4f}',
                'Pair(mix)': f'{pair_loss_mix.item():.4f}',
                'OT': f'{float(ot_loss):.4f}',
                'Emo': f'{emotion_loss.item():.4f}',
                'Cause': f'{cause_loss.item():.4f}',
                'LR': f'{current_lr:.6f}'
            }
            if getattr(args, 'use_ot_head', False) and outputs.get('ot_pair_scores') is not None:
                pf.update({'T-BCE': f'{float(transport_ce_loss):.4f}', 'T-KL': f'{float(distill_with_t_loss):.4f}'})
            progress_bar.set_postfix(pf)

    # Summary
    num_batches = len(dataloader)
    avg_pair_loss = total_pair_loss / num_batches
    avg_ot_loss = total_ot_loss / num_batches if getattr(args, 'use_ot_head', False) else 0.0
    avg_pair_mix = total_pair_mix / num_batches if getattr(args, 'use_ot_head', False) else avg_pair_loss
    avg_emotion_loss = total_emotion_loss / num_batches
    avg_cause_loss = total_cause_loss / num_batches
    avg_total_loss = total_loss / num_batches
    avg_transport_ce = (total_transport_ce / num_batches) if (getattr(args, 'use_ot_head', False)) else 0.0
    avg_distill = (total_distill / num_batches) if (getattr(args, 'use_ot_head', False)) else 0.0
    pair_precision, pair_recall, pair_f1 = MetricsCalculator.calculate_prf(
        torch.tensor(pair_preds), torch.tensor(pair_labels_all)
    )

    if len(emotion_preds) > 0:
        emotion_p, emotion_r, emotion_f1 = MetricsCalculator.calculate_prf(
            torch.tensor(emotion_preds), torch.tensor(emotion_trues),
            average='macro' if args.use_emocate else 'binary',
            use_emocate=args.use_emocate
        )
        cause_p, cause_r, cause_f1 = MetricsCalculator.calculate_prf(
            torch.tensor(cause_preds), torch.tensor(cause_trues)
        )
    else:
        emotion_p = emotion_r = emotion_f1 = 0.0
        cause_p = cause_r = cause_f1 = 0.0

    # PairEvaluator (pair extraction metrics)
    pred_pairs_by_conv = defaultdict(list)
    true_pairs_by_conv = defaultdict(list)
    for conv_id, emo_idx, cause_idx, pred_label, true_label in zip(
        convo_ids_all, emo_ids_all, cause_ids_all, pair_preds_all, pair_labels_all
    ):
        if true_label == 1:
            true_pairs_by_conv[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))
        if pred_label == 1:
            pred_pairs_by_conv[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))

    all_conv_ids = set(true_pairs_by_conv.keys()) | set(pred_pairs_by_conv.keys())
    all_true_pairs = [true_pairs_by_conv.get(conv_id, []) for conv_id in all_conv_ids]
    all_pred_pairs = [pred_pairs_by_conv.get(conv_id, []) for conv_id in all_conv_ids]
    pair_metrics = PairEvaluator.evaluate_pairs(all_true_pairs, all_pred_pairs) if all_conv_ids else {'precision': 0, 'recall': 0, 'f1': 0}

    # Top-k(T) decoding metrics (if exists)
    pair_metrics_topk = {'precision': 0, 'recall': 0, 'f1': 0}
    if pair_preds_all_topk:
        pred_pairs_by_conv_topk = defaultdict(list)
        true_pairs_by_conv_topk = defaultdict(list)
        for conv_id, emo_idx, cause_idx, pred_label, true_label in zip(
            convo_ids_all, emo_ids_all, cause_ids_all, pair_preds_all_topk, pair_labels_all
        ):
            if true_label == 1:
                true_pairs_by_conv_topk[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))
            if pred_label == 1:
                pred_pairs_by_conv_topk[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))

        all_conv_ids_topk = set(true_pairs_by_conv_topk.keys()) | set(pred_pairs_by_conv_topk.keys())
        all_true_pairs_topk = [true_pairs_by_conv_topk.get(conv_id, []) for conv_id in all_conv_ids_topk]
        all_pred_pairs_topk = [pred_pairs_by_conv_topk.get(conv_id, []) for conv_id in all_conv_ids_topk]
        pair_metrics_topk = PairEvaluator.evaluate_pairs(all_true_pairs_topk, all_pred_pairs_topk) if all_conv_ids_topk else {'precision': 0, 'recall': 0, 'f1': 0}

    pair_precision = pair_metrics.get('precision', 0.0)
    pair_recall = pair_metrics.get('recall', 0.0)
    pair_f1 = pair_metrics.get('f1', 0.0)

    if getattr(args, 'use_ot_head', False):
        logger.info(
            "Train - Loss(total/pair_mix/ot/pair_ce/T_BCE/T_KL/emotion/cause): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}, Pair P/R/F1: {:.4f}/{:.4f}/{:.4f}, Emotion P/R/F1: {:.4f}/{:.4f}/{:.4f}, Cause P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
                avg_total_loss, avg_pair_mix, avg_ot_loss, avg_pair_loss, avg_transport_ce, avg_distill, avg_emotion_loss, avg_cause_loss,
                pair_precision, pair_recall, pair_f1,
                emotion_p, emotion_r, emotion_f1,
                cause_p, cause_r, cause_f1
            )
        )
    else:
        logger.info(
            "Train - Loss(total/pair/emotion/cause): {:.4f}/{:.4f}/{:.4f}/{:.4f}, Pair P/R/F1: {:.4f}/{:.4f}/{:.4f}, Emotion P/R/F1: {:.4f}/{:.4f}/{:.4f}, Cause P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
                avg_total_loss, avg_pair_loss,  avg_emotion_loss, avg_cause_loss,
                emotion_p, emotion_r, emotion_f1,
                cause_p, cause_r, cause_f1
            )
        )

    return {
        'loss': avg_total_loss,
        'pair_loss': avg_pair_loss,
        'emotion_loss': avg_emotion_loss,
        'cause_loss': avg_cause_loss,
        'pair_precision': pair_precision,
        'pair_recall': pair_recall,
        'pair_f1': pair_f1,
        'pair_metrics': pair_metrics,
        'emotion_metrics': (emotion_p, emotion_r, emotion_f1),
        'cause_metrics': (cause_p, cause_r, cause_f1),
        't_topk_pair_metrics': pair_metrics_topk,
    }
def evaluate(model, dataloader, device, logger, args, pair_criterion=None):
    """Evaluation function"""
    model.eval()
    total_pair_loss = 0
    total_emotion_loss = 0
    total_cause_loss = 0
    total_loss = 0

    pair_probs_all = []
    pair_labels_all = []
    pair_preds_all = []
    pair_preds_all_topk = []
    convo_ids_all = []
    emo_ids_all = []
    cause_ids_all = []

    emotion_preds_all = []
    emotion_trues_all = []
    cause_preds_all = []
    cause_trues_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            doc_len = batch['doc_len'].to(device)
            speakers = batch['speakers'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            cause_labels = batch['cause_labels'].to(device)
            pair_indices = batch['pair_indices'].to(device)
            pair_distances = batch['pair_distances'].to(device)
            pair_labels = batch['pair_labels'].to(device)

            precomputed_features = batch.get('precomputed_features', None)
            if precomputed_features is not None:
                precomputed_features = precomputed_features.to(device).float()
            outputs = model(
                precomputed_features,
                doc_len, speakers,
                pair_indices, pair_distances
            )

            pair_logits = outputs['pair_logits']
            emotion_logits = outputs['emotion_logits']
            cause_logits = outputs['cause_logits']

  
            doc_mask = MaskGenerator.create_padding_mask(doc_len, speakers.size(1)).to(device)
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

            # Pair loss
            if pair_criterion is None:
                pair_loss = F.cross_entropy(pair_logits, pair_labels)
            else:
                pair_loss = pair_criterion(pair_logits, pair_labels)

            loss = pair_loss + args.emotion_weight * emotion_loss + args.cause_weight * cause_loss
            total_pair_loss += pair_loss.item()
            total_emotion_loss += emotion_loss.item()
            total_cause_loss += cause_loss.item()
            total_loss += loss.item()

            T_eval = 1.0
            probs = torch.softmax(pair_logits / T_eval, dim=-1)[:, 1]

            if getattr(args, 'use_ot_head', False) and hasattr(model, 'ot_head') and model.ot_head is not None:
                fgw_scores = outputs.get('ot_pair_scores')
                if fgw_scores is None:
                    fgw_scores, _ = model.ot_head(
                        outputs['emotion_context'], outputs['cause_context'],
                        outputs['edge_weights_e'], outputs['edge_weights_c'],
                        pair_indices, doc_len,
                        pred_future_cause=getattr(args, 'pred_future_cause', False),
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

            pair_probs_all.extend(probs.cpu().tolist())
            pair_labels_all.extend(pair_labels.cpu().tolist())
       
            # Global MCMF decoding
            preds = global_mcmf_decode(
                    pair_indices=pair_indices,
                    probs=probs,
                    conv_ids=batch['pair_conversation_id'],
                    row_cap=getattr(args, 'mcmf_row_capacity', 1),
                    col_cap=getattr(args, 'mcmf_col_capacity', 1),
                    lambda_cost=getattr(args, 'mcmf_lambda', 0.5),
                    score_space=getattr(args, 'mcmf_score_space', 'prob'),
                    eps=getattr(args, 'mcmf_eps', 1e-6),
                    pre_topk_per_row=getattr(args, 'mcmf_pre_topk_per_row', None),
                    pre_min_prob=getattr(args, 'mcmf_pre_min_prob', None),
                    pre_min_logit=getattr(args, 'mcmf_pre_min_logit', None),
                    device=device,
                )
            # Length alignment protection (after final decoding)
            expected_pairs = pair_indices.size(0)
            if preds is None or preds.numel() != expected_pairs:
                logger.warning(f'Preds length mismatch(after decode): got {0 if preds is None else preds.numel()}, expected {expected_pairs}. Fallback to zeros.')
                preds = torch.zeros(expected_pairs, dtype=torch.long, device=device)

            pair_preds_all.extend(preds.detach().cpu().tolist())
            
            # T-based top-k soft capacity decoding (evaluation phase)
            if getattr(args, 'use_ot_head', False):
                preds_topk = decode_transport_topk(
                    outputs, pair_indices, batch['pair_conversation_id'],
                    getattr(args, 'mcmf_row_capacity', 1),
                    getattr(args, 'fgw_row_temp', 0.7),
                    device
                )
                if preds_topk is not None:
                    pair_preds_all_topk.extend(preds_topk.detach().cpu().tolist())
            
            convo_ids_all.extend(batch['pair_conversation_id'])
            emo_ids_all.extend(pair_indices[:, 1].cpu().tolist())
            cause_ids_all.extend(pair_indices[:, 2].cpu().tolist())


            emo_pred = torch.argmax(emotion_logits, dim=-1)
            cause_pred = torch.argmax(cause_logits, dim=-1)
            mask_cpu = doc_mask.cpu()
            emotion_preds_all.extend(emo_pred.cpu()[mask_cpu].tolist())
            emotion_trues_all.extend(emotion_labels.cpu()[mask_cpu].tolist())
            cause_preds_all.extend(cause_pred.cpu()[mask_cpu].tolist())
            cause_trues_all.extend(cause_labels.cpu()[mask_cpu].tolist())

    num_batches = len(dataloader)
    # Use max(1, num_batches) to prevent division by zero in extreme cases
    denom = max(1, num_batches)
    avg_pair_loss = total_pair_loss / denom
    avg_emotion_loss = total_emotion_loss / denom
    avg_cause_loss = total_cause_loss / denom
    avg_total_loss = total_loss / denom

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

    # PairEvaluator (pair extraction metrics)
    pred_pairs_by_conv = defaultdict(list)
    true_pairs_by_conv = defaultdict(list)
    for conv_id, emo_idx, cause_idx, pred_label, true_label in zip(
        convo_ids_all, emo_ids_all, cause_ids_all, pair_preds_all, pair_labels_all
    ):
        if true_label == 1:
            true_pairs_by_conv[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))
        if pred_label == 1:
            pred_pairs_by_conv[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))

    all_conv_ids = set(true_pairs_by_conv.keys()) | set(pred_pairs_by_conv.keys())
    all_true_pairs = [true_pairs_by_conv.get(conv_id, []) for conv_id in all_conv_ids]
    all_pred_pairs = [pred_pairs_by_conv.get(conv_id, []) for conv_id in all_conv_ids]
    pair_metrics = PairEvaluator.evaluate_pairs(all_true_pairs, all_pred_pairs) if all_conv_ids else {'precision': 0, 'recall': 0, 'f1': 0}

    # T-based top-k decoding metrics (if available)
    pair_metrics_topk = {'precision': 0, 'recall': 0, 'f1': 0}
    if pair_preds_all_topk:
        pred_pairs_by_conv_topk = defaultdict(list)
        true_pairs_by_conv_topk = defaultdict(list)
        for conv_id, emo_idx, cause_idx, pred_label, true_label in zip(
            convo_ids_all, emo_ids_all, cause_ids_all, pair_preds_all_topk, pair_labels_all
        ):
            if true_label == 1:
                true_pairs_by_conv_topk[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))
            if pred_label == 1:
                pred_pairs_by_conv_topk[conv_id].append((conv_id, emo_idx + 1, cause_idx + 1))

        all_conv_ids_topk = set(true_pairs_by_conv_topk.keys()) | set(pred_pairs_by_conv_topk.keys())
        all_true_pairs_topk = [true_pairs_by_conv_topk.get(conv_id, []) for conv_id in all_conv_ids_topk]
        all_pred_pairs_topk = [pred_pairs_by_conv_topk.get(conv_id, []) for conv_id in all_conv_ids_topk]
        pair_metrics_topk = PairEvaluator.evaluate_pairs(all_true_pairs_topk, all_pred_pairs_topk) if all_conv_ids_topk else {'precision': 0, 'recall': 0, 'f1': 0}

    # Output evaluation metrics
    logger.info(
        "Eval - Loss(total/pair/emotion/cause): {:.4f}/{:.4f}/{:.4f}/{:.4f}, Pair P/R/F1: {:.4f}/{:.4f}/{:.4f}, Emotion P/R/F1: {:.4f}/{:.4f}/{:.4f}, Cause P/R/F1: {:.4f}/{:.4f}/{:.4f}".format(
            avg_total_loss, avg_pair_loss, avg_emotion_loss, avg_cause_loss,
            pair_precision, pair_recall, pair_f1,
            emotion_p, emotion_r, emotion_f1,
            cause_p, cause_r, cause_f1
        )
    )

    pair_precision = pair_metrics.get('precision', 0.0)
    pair_recall = pair_metrics.get('recall', 0.0)
    pair_f1 = pair_metrics.get('f1', 0.0)

    return {
        'loss': avg_total_loss,
        'pair_loss': avg_pair_loss,
        'emotion_loss': avg_emotion_loss,
        'cause_loss': avg_cause_loss,
        'pair_precision': pair_precision,
        'pair_recall': pair_recall,
        'pair_f1': pair_f1,
        'pair_metrics': pair_metrics,
        'emotion_metrics': (emotion_p, emotion_r, emotion_f1),
        'cause_metrics': (cause_p, cause_r, cause_f1),
        't_topk_pair_metrics': pair_metrics_topk,
        'pair_probs': pair_probs_all,
        'pair_labels': pair_labels_all,
        'pair_preds': pair_preds_all,
        'convo_ids': convo_ids_all,
        'emo_ids': emo_ids_all,
        'cause_ids': cause_ids_all
    }


def main():

    config = Config()
    args = config.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    logger = setup_logging(args.log_dir, args.dataset, 'ecpec')
    print_time()
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    logger.info("Loading data...")
    train_loader, test_loader, dev_loader, _ = create_feature_data_loaders(
        dataset_name=args.dataset,
        feature_dir=getattr(args, 'feature_dir', './features'),
        batch_size=args.batch_size,
        pred_future_cause=args.pred_future_cause,
        use_emocate=args.use_emocate,
        eval_max_pair_distance=getattr(args, 'eval_max_pair_distance', None),
        train_max_pair_distance=getattr(args, 'train_max_pair_distance', None),
        max_doc_len=args.max_doc_len
    )

    logger.info("Building model...")
    model_config = get_model_config(args)
    model = ECPEC_Model(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")


    # Use AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Use adaptive learning rate based on validation metrics (ReduceLROnPlateau)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',            # Monitor F1, higher is better
        factor=getattr(args, 'plateau_factor', 0.5),
        patience=getattr(args, 'plateau_patience', 2),
        threshold=getattr(args, 'plateau_threshold', 1e-3),
        threshold_mode='rel',
        cooldown=getattr(args, 'plateau_cooldown', 1),
        min_lr=getattr(args, 'plateau_min_lr', 1e-6),
        verbose=True
    )

    weight_ratio_cap = getattr(args, 'weight_ratio_cap', 5.0) 
    class_weights = PairLoss.compute_class_weights(train_loader, device, weight_ratio_cap)
    criterion = PairLoss(class_weight=class_weights)
    logger.info(f"category weights: {class_weights}")

    early_stopping = EarlyStopping(patience=args.patience)
    model_saver = ModelSaver(args.save_dir)


    start_epoch = 0
    best_f1 = 0
    if args.load_checkpoint:
        start_epoch, metrics = model_saver.load_checkpoint(model, optimizer, args.load_checkpoint)
        if metrics:
            best_f1 = metrics.get('best_f1', 0)
            logger.info(f"Restored from checkpoint: start_epoch={start_epoch}, best_f1={best_f1}")

    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, device, logger, args, pair_criterion=criterion)


        if (epoch + 1) % args.eval_interval == 0:
            if dev_loader:
                eval_metrics = evaluate(model, dev_loader, device, logger, args, pair_criterion=criterion)
                eval_f1 = eval_metrics['pair_f1']
            else:
                eval_metrics = evaluate(model, test_loader, device, logger, args, pair_criterion=criterion)
                eval_f1 = eval_metrics['pair_f1'] 

            # Learning rate scheduling: drive ReduceLROnPlateau with validation Pair F1
            try:
                scheduler.step(eval_f1)
                logger.info(f"Scheduler on plateau step with eval_f1={eval_f1:.4f}, current LR={optimizer.param_groups[0]['lr']:.6f}")
            except Exception as e:
                logger.warning(f"Scheduler step failed: {e}")

            # Check if refreshing best
            is_best = eval_f1 > best_f1
            if is_best:
                best_f1 = eval_f1
                logger.info(f"Current best F1: {best_f1:.4f}")

 
            if (epoch + 1) % args.save_interval == 0 or is_best:
                metrics = {
                    'epoch': epoch + 1,
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'best_f1': best_f1
                }
                model_saver.save_checkpoint(model, optimizer, epoch + 1, metrics, is_best)

            # Unified F1 maximization objective: early stopping monitors -F1
            val_score = eval_metrics['pair_f1']
            if early_stopping(-val_score, model):
                logger.info("Early stopping triggered")
                break


    logger.info("\n=== Training Complete ===")
    model_saver.load_checkpoint(model, filename='best_model.pt')
    logger.info("\n--- Test Evaluation ---")
    test_metrics_default = evaluate(model, test_loader, device, logger, args, pair_criterion=criterion)



if __name__ == "__main__":
    main()
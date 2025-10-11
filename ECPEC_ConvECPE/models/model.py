# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    IdentityTextEncoder,
    GraphConversationEncoder,
    MaskGenerator
)
from alignment.fgw_torch import DifferentiableFGWHead


class Model(nn.Module):
    """端到端多任务模型：整段对话输入，联合训练情感、原因检测与情感-原因对分类。"""

    def __init__(self, config):
        super().__init__()

        if config.model_type != 'bert':
            raise ValueError("当前端到端实现仅支持BERT文本编码 (model_type='bert')")

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.n_emotions = config.n_emotions
        self.dropout = nn.Dropout(config.dropout)
        # 仅保留预计算特征路径
        self.text_encoder = IdentityTextEncoder()
        self.use_doc_encoder = False

        # Node-side additive injection: absolute position + speaker embeddings
        self.use_pos_embed = getattr(config, 'use_pos_embed', True)
        self.use_speaker_embed = getattr(config, 'use_speaker_embed', True)
        pos_dim = getattr(config, 'position_embedding_dim', 64)
        max_doc_len = getattr(config, 'max_doc_len', 50)
        self.abs_pos_embed = nn.Embedding(max_doc_len, pos_dim)
        self.speaker_vocab_size = getattr(config, 'speaker_vocab_size', 16)
        self.speaker_embed = nn.Embedding(self.speaker_vocab_size, pos_dim)
        self.pos_proj = nn.Linear(pos_dim, self.hidden_dim) if pos_dim != self.hidden_dim else nn.Identity()
        self.spk_proj = nn.Linear(pos_dim, self.hidden_dim) if pos_dim != self.hidden_dim else nn.Identity()
        self.node_ln = nn.LayerNorm(self.hidden_dim)
        self.node_dropout = nn.Dropout(config.dropout)


        graph_kwargs = dict(
            hidden_dim=self.hidden_dim,
            num_layers=getattr(config, 'graph_num_layers', 2),
            num_heads=config.n_heads,
            dropout=config.dropout,
            window_size=getattr(config, 'graph_window_size', 3),
            use_speaker_edges=getattr(config, 'use_speaker_edges', True),
            use_temporal_edges=getattr(config, 'use_temporal_edges', True),
            distance_decay=getattr(config, 'distance_decay', True),
            tau=getattr(config, 'graph_tau', 2.0)
        )
        self.emotion_graph_encoder = GraphConversationEncoder(**graph_kwargs)
        self.cause_graph_encoder = GraphConversationEncoder(**graph_kwargs)

        self.emotion_classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.n_emotions)
        )
        self.cause_classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, 2)
        )

        # FGW-only 开关：为 True 时不构建对级 MLP 及其相关嵌入
        self.fgw_only = getattr(config, 'fgw_only', False)
        if not self.fgw_only:
            self.emotion_category_embedding = nn.Embedding(
                self.n_emotions, config.position_embedding_dim
            )
            self.distance_offset = 200
            self.distance_embedding = nn.Embedding(401, config.position_embedding_dim)

            pair_feature_dim = self.hidden_dim * 2
            pair_feature_dim += 2 * config.position_embedding_dim

            self.pair_classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(pair_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.hidden_dim, 2)
            )
        else:
            # 保持属性存在性，便于 forward 条件分支
            self.emotion_category_embedding = None
            self.distance_offset = 200
            self.distance_embedding = None
            self.pair_classifier = None

        self.use_ot_head = getattr(config, 'use_ot_head', False)
        if self.use_ot_head:
            self.ot_head = DifferentiableFGWHead(
                alpha=getattr(config, 'fgw_alpha', 0.5),
                eps=getattr(config, 'fgw_eps', 0.1),
                max_iter=getattr(config, 'fgw_iterations', 5),
                sinkhorn_iter=getattr(config, 'fgw_sinkhorn_iter', 20),
                sinkhorn_eps=getattr(config, 'fgw_sinkhorn_eps', 1e-6),
                row_norm=getattr(config, 'fgw_row_norm', 'row_softmax'),
                row_temp=getattr(config, 'fgw_row_temp', 0.7)
            )
        else:
            self.ot_head = None


    def forward(self, input_ids, attention_mask,
                doc_lengths, speakers, pair_indices, pair_distances, texts=None, precomputed_features=None):
        # Select device from speakers tensor
        device = speakers.device

        if precomputed_features is None:
            raise ValueError("本模型已移除端到端文本编码路径，请提供 precomputed_features=(B,L,D)")
        utterance_repr = precomputed_features  # (B, L, D)
        batch_size, max_doc_len = utterance_repr.shape[:2]

        # Node-side additive injection: absolute position + speaker embeddings
        x = utterance_repr
        if self.use_pos_embed:
            pos_ids = torch.arange(max_doc_len, device=device)
            pos_ids = pos_ids.clamp_max(self.abs_pos_embed.num_embeddings - 1)
            pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
            pos_feat = self.abs_pos_embed(pos_ids)
            pos_feat = self.pos_proj(pos_feat)
            x = x + pos_feat
        if self.use_speaker_embed:
            spk_ids = speakers.clamp(min=0, max=self.speaker_vocab_size - 1)
            spk_feat = self.speaker_embed(spk_ids)
            spk_feat = self.spk_proj(spk_feat)
            x = x + spk_feat
        x = self.node_ln(x)
        x = self.node_dropout(x)

        doc_mask = MaskGenerator.create_padding_mask(doc_lengths, max_doc_len).to(device)
        emotion_context = self.emotion_graph_encoder(x, speakers, doc_lengths, doc_mask)
        cause_context = self.cause_graph_encoder(x, speakers, doc_lengths, doc_mask)

        adjacency, edge_weights = self.emotion_graph_encoder.adjacency_builder.build_adjacency(
            speakers, doc_lengths, max_doc_len,
            window_size=self.config.graph_window_size,
            use_speaker_edges=self.config.use_speaker_edges,
            use_temporal_edges=self.config.use_temporal_edges,
            distance_decay=self.config.distance_decay,
            tau=self.config.graph_tau
        )

        emotion_logits = self.emotion_classifier(emotion_context)
        cause_logits = self.cause_classifier(cause_context)

        pair_logits = None
        if not self.fgw_only:
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            emotion_category_feat = torch.matmul(
                emotion_probs, self.emotion_category_embedding.weight
            )

            batch_idx = pair_indices[:, 0]
            emo_idx = pair_indices[:, 1]
            cause_idx = pair_indices[:, 2]

            emo_vec = emotion_context[batch_idx, emo_idx]
            cause_vec = cause_context[batch_idx, cause_idx]
            pair_features = [emo_vec, cause_vec]

            distance_indices = torch.clamp(
                pair_distances + self.distance_offset,
                0, self.distance_embedding.num_embeddings - 1
            )
            distance_embed = self.distance_embedding(distance_indices)
            pair_features.append(distance_embed)

            emotion_cat_embed = emotion_category_feat[batch_idx, emo_idx]
            pair_features.append(emotion_cat_embed)

            pair_features = torch.cat(pair_features, dim=-1)
            pair_logits = self.pair_classifier(pair_features)

        ot_pair_scores = None
        ot_transports = None
        if self.ot_head is not None:
            ot_pair_scores, ot_transports = self.ot_head(
                emotion_context, cause_context,
                edge_weights,
                pair_indices,
                doc_lengths,
                pred_future_cause=getattr(self.config, 'pred_future_cause', True),
                max_pair_distance=getattr(self.config, 'train_max_pair_distance', None)
            )

        return {
            'pair_logits': pair_logits,
            'emotion_logits': emotion_logits,
            'cause_logits': cause_logits,
            'emotion_context': emotion_context,
            'cause_context': cause_context,
            'ot_pair_scores': ot_pair_scores,
            'ot_transports': ot_transports,
            'edge_weights': edge_weights
        }


class ModelConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.get('model_type', 'bert')
        self.hidden_dim = kwargs.get('hidden_dim', 200)
        self.dropout = kwargs.get('dropout', 0.1)
        self.vocab_size = kwargs.get('vocab_size', 30000)
        self.embedding_dim = kwargs.get('embedding_dim', 300)
        self.position_embedding_dim = kwargs.get('position_embedding_dim', 50)
        self.max_sen_len = kwargs.get('max_sen_len', 35)
        self.max_doc_len = kwargs.get('max_doc_len', 50)
        self.n_heads = kwargs.get('n_heads', 8)
        self.graph_num_layers = kwargs.get('graph_num_layers', 2)
        self.graph_window_size = kwargs.get('graph_window_size', 3)
        self.use_speaker_edges = kwargs.get('use_speaker_edges', True)
        self.use_temporal_edges = kwargs.get('use_temporal_edges', True)
        self.distance_decay = kwargs.get('distance_decay', True)
        self.graph_tau = kwargs.get('graph_tau', 2.0)
        self.use_emotion_category = kwargs.get('use_emotion_category', True)
        self.n_emotions = kwargs.get('n_emotions', 7)

        # 可微OT头与FGW相关参数
        self.use_ot_head = kwargs.get('use_ot_head', False)
        self.fgw_alpha = kwargs.get('fgw_alpha', 0.5)
        self.fgw_eps = kwargs.get('fgw_eps', 0.1)
        self.fgw_iterations = kwargs.get('fgw_iterations', 5)
        self.fgw_sinkhorn_iter = kwargs.get('fgw_sinkhorn_iter', 20)
        self.fgw_sinkhorn_eps = kwargs.get('fgw_sinkhorn_eps', 1e-6)
        # FGW prior controls
        self.pred_future_cause = kwargs.get('pred_future_cause', True)
        self.train_max_pair_distance = kwargs.get('train_max_pair_distance', None)
        # FGW-only
        self.fgw_only = kwargs.get('fgw_only', False)


class PairLoss(nn.Module):
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



# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    GraphConversationEncoderRelational,
    MaskGenerator,
)
from alignment.fgw_torch import DifferentiableFGWHead


class PairBiaffineHead(nn.Module):
    """Pair-level biaffine head with gated interaction."""

    def __init__(self, hidden_dim, pos_dim, use_gate=True, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.use_gate = use_gate

        self.dropout = nn.Dropout(dropout)
        self.feature_dropout = nn.Dropout(dropout)
        self.emotion_norm = nn.LayerNorm(hidden_dim)
        self.cause_norm = nn.LayerNorm(hidden_dim)

        # Bilinear term: one weight matrix per class
        self.W = nn.Parameter(torch.empty(2, hidden_dim, hidden_dim))

        # Linear features: [e; c; |e-c|; e∘c; dist; emo_cat]
        feature_dim = 4 * hidden_dim + 2 * pos_dim
        self.linear = nn.Linear(feature_dim, 2)

        if use_gate:
            gate_input_dim = 2 * hidden_dim + 2 * pos_dim
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, 2 * hidden_dim),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if self.gate is not None:
            for module in self.gate:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, emotion_vec, cause_vec, distance_embed, emotion_cat_embed):

        e = self.emotion_norm(emotion_vec)
        c = self.cause_norm(cause_vec)

        e = self.dropout(e)
        c = self.dropout(c)

        diff = torch.abs(e - c)
        prod = e * c

        if self.gate is not None:
            gate_input = torch.cat([e, c, distance_embed, emotion_cat_embed], dim=-1)
            gate_values = self.gate(gate_input)
            g_diff, g_prod = torch.split(gate_values, self.hidden_dim, dim=-1)
            diff = diff * g_diff
            prod = prod * g_prod

        linear_features = torch.cat(
            [e, c, diff, prod, distance_embed, emotion_cat_embed],
            dim=-1
        )
        linear_features = self.feature_dropout(linear_features)
        linear_logits = self.linear(linear_features)

        bilinear_logits = torch.einsum('bd,kdm,bm->bk', e, self.W, c)

        return linear_logits + bilinear_logits


class Model(nn.Module):
    """End-to-end multi-task model"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.n_emotions = config.n_emotions
        
        # Node-level embeddings
        self.use_pos_embed = config.use_pos_embed
        self.use_speaker_embed = config.use_speaker_embed
        pos_dim = config.position_embedding_dim
        max_doc_len = config.max_doc_len
        self.abs_pos_embed = nn.Embedding(max_doc_len, pos_dim)
        self.speaker_vocab_size = config.speaker_vocab_size
        self.speaker_embed = nn.Embedding(self.speaker_vocab_size, pos_dim)
        self.pos_proj = nn.Linear(pos_dim, self.hidden_dim) if pos_dim != self.hidden_dim else nn.Identity()
        self.spk_proj = nn.Linear(pos_dim, self.hidden_dim) if pos_dim != self.hidden_dim else nn.Identity()
        self.node_ln = nn.LayerNorm(self.hidden_dim)
        self.node_dropout = nn.Dropout(config.dropout)


        graph_kwargs_rel = dict(
            hidden_dim=self.hidden_dim,
            num_layers=getattr(config, 'graph_num_layers', 2),
            num_heads=config.n_heads,
            dropout=config.dropout,
            attn_dropout=config.dropout,
            ffn_dropout=config.dropout,
            window_size=getattr(config, 'graph_window_size', 3),
            use_speaker_edges=getattr(config, 'use_speaker_edges', True),
            use_temporal_edges=getattr(config, 'use_temporal_edges', True),
            distance_decay=getattr(config, 'distance_decay', True),
            tau=getattr(config, 'graph_tau', 2.0),
            rel_num_bases=getattr(config, 'rel_num_bases', 4),
            rel_use_knn=getattr(config, 'rel_use_knn', True),
            rel_knn_k=getattr(config, 'rel_knn_k', 6),
            rel_knn_min_sim=getattr(config, 'rel_knn_min_sim', 0.5),
            rel_edge_drop=getattr(config, 'rel_edge_drop', 0.1),
        )
        self.emotion_graph_encoder = GraphConversationEncoderRelational(**graph_kwargs_rel)
        self.cause_graph_encoder = GraphConversationEncoderRelational(**graph_kwargs_rel)

        self.emotion_classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.n_emotions)
        )
        self.cause_classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, 2)
        )

        self.emotion_category_embedding = nn.Embedding(
            self.n_emotions, config.position_embedding_dim
        )
        self.distance_offset = 200
        self.distance_embedding = nn.Embedding(401, config.position_embedding_dim)
        self.pair_head = PairBiaffineHead(
            hidden_dim=self.hidden_dim,
            pos_dim=config.position_embedding_dim,
            use_gate=True,
            dropout=config.dropout
        )

        self.use_ot_head = getattr(config, 'use_ot_head', False)
        if self.use_ot_head:
            self.ot_head = DifferentiableFGWHead(
                alpha=getattr(config, 'fgw_alpha', 0.5),
                eps=getattr(config, 'fgw_eps', 0.1),
                max_iter=getattr(config, 'fgw_iterations', 5),
                sinkhorn_iter=getattr(config, 'fgw_sinkhorn_iter', 20),
                sinkhorn_eps=getattr(config, 'fgw_sinkhorn_eps', 1e-6),
                row_norm=getattr(config, 'fgw_row_norm', 'entmax'),
                row_temp=getattr(config, 'fgw_row_temp', 1.0),
                entmax_alpha=getattr(config, 'fgw_entmax_alpha', 1.5),
                entmax_bisect_iter=getattr(config, 'fgw_entmax_bisect_iter', 50),
                use_uot=getattr(config, 'fgw_use_uot', False),
                uot_rho=getattr(config, 'fgw_uot_rho', 1.0),
                attr_metric=getattr(config, 'fgw_attr_metric', 'maha'),
                feat_dim=self.hidden_dim
            )
        else:
            self.ot_head = None


    def forward(self, precomputed_features, doc_lengths, speakers, pair_indices, pair_distances):

        device = speakers.device

        if precomputed_features is None:
            raise ValueError("precomputed_features is required (B, L, D)")
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

        # Build relational adjacency for FGW structure term.
        # We construct two graphs (emotion-side and cause-side). They share the same rule-based
        # edges (self/temporal/speaker) but may differ in KNN edges because KNN depends on node features.
        _, _, _, edge_weights_e = self.emotion_graph_encoder.adjacency_builder.build_relational_adjacency(
            speakers, doc_lengths, max_doc_len,
            window_size=self.config.graph_window_size,
            use_speaker_edges=self.config.use_speaker_edges,
            use_temporal_edges=self.config.use_temporal_edges,
            distance_decay=self.config.distance_decay,
            tau=self.config.graph_tau,
            node_features=emotion_context,
            use_knn=getattr(self.config, 'rel_use_knn', True),
            knn_k=getattr(self.config, 'rel_knn_k', 6),
            knn_min_sim=getattr(self.config, 'rel_knn_min_sim', 0.5),
        )
        _, _, _, edge_weights_c = self.cause_graph_encoder.adjacency_builder.build_relational_adjacency(
            speakers, doc_lengths, max_doc_len,
            window_size=self.config.graph_window_size,
            use_speaker_edges=self.config.use_speaker_edges,
            use_temporal_edges=self.config.use_temporal_edges,
            distance_decay=self.config.distance_decay,
            tau=self.config.graph_tau,
            node_features=cause_context,
            use_knn=getattr(self.config, 'rel_use_knn', True),
            knn_k=getattr(self.config, 'rel_knn_k', 6),
            knn_min_sim=getattr(self.config, 'rel_knn_min_sim', 0.5),
        )

        emotion_logits = self.emotion_classifier(emotion_context)
        cause_logits = self.cause_classifier(cause_context)

        # Compute pair logits using biaffine head
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        emotion_category_feat = torch.matmul(
            emotion_probs, self.emotion_category_embedding.weight
        )

        batch_idx = pair_indices[:, 0]
        emo_idx = pair_indices[:, 1]
        cause_idx = pair_indices[:, 2]

        emo_vec = emotion_context[batch_idx, emo_idx]
        cause_vec = cause_context[batch_idx, cause_idx]
        distance_indices = torch.clamp(
            pair_distances + self.distance_offset,
            0, self.distance_embedding.num_embeddings - 1
        )
        distance_embed = self.distance_embedding(distance_indices)
        emotion_cat_embed = emotion_category_feat[batch_idx, emo_idx]

        pair_logits = self.pair_head(
            emo_vec, cause_vec, distance_embed, emotion_cat_embed
        )

        ot_pair_scores = None
        ot_transports = None
        if self.ot_head is not None:
            ot_pair_scores, ot_transports = self.ot_head(
                emotion_context, cause_context,
                edge_weights_e, edge_weights_c,
                pair_indices,
                doc_lengths,
                pred_future_cause=getattr(self.config, 'pred_future_cause', False),
                max_pair_distance=getattr(self.config, 'train_max_pair_distance', None),
                row_cap=getattr(self.config, 'mcmf_row_capacity', None),
                col_cap=getattr(self.config, 'mcmf_col_capacity', None)
            )

        return {
            'pair_logits': pair_logits,
            'emotion_logits': emotion_logits,
            'cause_logits': cause_logits,
            'emotion_context': emotion_context,
            'cause_context': cause_context,
            'ot_pair_scores': ot_pair_scores,
            'ot_transports': ot_transports,
            # Keep both graphs for analysis/debugging and for any downstream use.
            'edge_weights_e': edge_weights_e,
            'edge_weights_c': edge_weights_c,
            # Backward compatibility: keep the old key pointing to emotion-side graph.
            'edge_weights': edge_weights_e,
        }


class ModelConfig:
    def __init__(self, **kwargs):
        self.hidden_dim = kwargs.get('hidden_dim', 200)
        self.dropout = kwargs.get('dropout', 0.3)       
        self.position_embedding_dim = kwargs.get('position_embedding_dim', 50)
        self.max_doc_len = kwargs.get('max_doc_len', 50)
        self.n_heads = kwargs.get('n_heads', 8)
        self.use_pos_embed = kwargs.get('use_pos_embed', True)
        self.use_speaker_embed = kwargs.get('use_speaker_embed', True)
        self.speaker_vocab_size = kwargs.get('speaker_vocab_size', 16)
        self.graph_num_layers = kwargs.get('graph_num_layers', 2)
        self.graph_window_size = kwargs.get('graph_window_size', 3)
        self.use_speaker_edges = kwargs.get('use_speaker_edges', True)
        self.use_temporal_edges = kwargs.get('use_temporal_edges', True)
        self.distance_decay = kwargs.get('distance_decay', True)
        self.graph_tau = kwargs.get('graph_tau', 2.0)
        self.n_emotions = kwargs.get('n_emotions', 7)
        self.use_ot_head = kwargs.get('use_ot_head', False)
        self.fgw_alpha = kwargs.get('fgw_alpha', 0.5)
        self.fgw_eps = kwargs.get('fgw_eps', 0.1)
        self.fgw_iterations = kwargs.get('fgw_iterations', 5)
        self.fgw_sinkhorn_iter = kwargs.get('fgw_sinkhorn_iter', 20)
        self.fgw_sinkhorn_eps = kwargs.get('fgw_sinkhorn_eps', 1e-6)
        self.fgw_row_norm = kwargs.get('fgw_row_norm', 'entmax')
        self.fgw_row_temp = kwargs.get('fgw_row_temp', 1.0)
        self.fgw_entmax_alpha = kwargs.get('fgw_entmax_alpha', 1.5)
        self.fgw_entmax_bisect_iter = kwargs.get('fgw_entmax_bisect_iter', 50)       
        self.fgw_use_uot = kwargs.get('fgw_use_uot', False)
        self.fgw_uot_rho = kwargs.get('fgw_uot_rho', 1.0)
        self.pred_future_cause = kwargs.get('pred_future_cause', False)
        self.train_max_pair_distance = kwargs.get('train_max_pair_distance', None)
        self.mcmf_row_capacity = kwargs.get('mcmf_row_capacity', None)
        self.mcmf_col_capacity = kwargs.get('mcmf_col_capacity', None)
        self.rel_num_bases = kwargs.get('rel_num_bases', 4)
        self.rel_use_knn = kwargs.get('rel_use_knn', True)
        self.rel_knn_k = kwargs.get('rel_knn_k', 6)
        self.rel_knn_min_sim = kwargs.get('rel_knn_min_sim', 0.5)
        self.rel_edge_drop = kwargs.get('rel_edge_drop', 0.1)
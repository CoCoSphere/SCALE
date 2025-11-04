# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGenerator(nn.Module):
    """Mask generator"""

    @staticmethod
    def create_padding_mask(lengths, max_len):
        """Create padding mask"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask

    @staticmethod
    def create_causal_mask(seq_len):
        """Create causal mask (lower triangular matrix)"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.bool()


class GraphAdjacencyBuilder:
    """Graph adjacency matrix builder"""

    @staticmethod
    def build_adjacency(speakers, doc_lengths, max_doc_len, window_size=3,
                       use_speaker_edges=True, use_temporal_edges=True,
                       distance_decay=True, tau=2.0):

        batch_size = speakers.size(0)
        device = speakers.device
        L = max_doc_len

        # Initialize adjacency and weight matrices
        adjacency = torch.zeros(batch_size, L, L, device=device)
        edge_weights = torch.zeros(batch_size, L, L, device=device)

        # Valid position mask (by actual conversation length)
        idx = torch.arange(L, device=device)
        valid = idx.unsqueeze(0) < doc_lengths.unsqueeze(1)  # (B, L)
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)  # (B, L, L)


        I = torch.eye(L, device=device)
        adjacency = adjacency + (I.unsqueeze(0) * pair_valid.float())
        edge_weights = edge_weights + (I.unsqueeze(0) * pair_valid.float())

        # Distance matrix (for temporal and speaker edge weights)
        dist = torch.abs(idx.view(1, L) - idx.view(L, 1))  # (L, L)
        dist_b = dist.unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, L)

        # Temporal adjacency (off-diagonal, within window)
        if use_temporal_edges:
            temporal_mask = (dist_b > 0) & (dist_b <= window_size) & pair_valid  # (B, L, L)
            if distance_decay:
                temp_w = torch.exp(-dist_b.float() / float(tau))
            else:
                temp_w = torch.ones_like(dist_b, dtype=torch.float32)
            adjacency[temporal_mask] = 1.0
            edge_weights[temporal_mask] = temp_w[temporal_mask]

        if use_speaker_edges:
            same_speaker = speakers.unsqueeze(2).eq(speakers.unsqueeze(1))  # (B, L, L)
            speaker_mask = (dist_b > 0) & (dist_b <= window_size) & same_speaker & pair_valid
            if distance_decay:
                sp_w = torch.exp(-dist_b.float() / float(tau)) + 0.5
            else:
                sp_w = torch.full_like(dist_b, 1.5, dtype=torch.float32)
            adjacency[speaker_mask] = 1.0
            edge_weights[speaker_mask] = sp_w[speaker_mask]

        return adjacency, edge_weights

    @staticmethod
    def build_relational_adjacency(
        speakers,
        doc_lengths,
        max_doc_len,
        window_size=3,
        use_speaker_edges=True,
        use_temporal_edges=True,
        distance_decay=True,
        tau=2.0,
        node_features=None,
        use_knn=True,
        knn_k=6,
        knn_min_sim=0.5,
    ):
       
        device = speakers.device
        batch_size = speakers.size(0)
        L = max_doc_len
        R = 5

        rel_masks = torch.zeros(batch_size, R, L, L, device=device, dtype=torch.bool)
        rel_weights = torch.zeros(batch_size, R, L, L, device=device)

        idx = torch.arange(L, device=device)
        valid = idx.unsqueeze(0) < doc_lengths.unsqueeze(1)  # (B,L)
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)

        # self
        I = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
        rel_masks[:, 0] = I & pair_valid
        rel_weights[:, 0] = I.float()

        # temporal
        dist = torch.abs(idx.view(1, L) - idx.view(L, 1))
        dist_b = dist.unsqueeze(0).expand(batch_size, -1, -1)
        fwd = (idx.view(1, 1, L) < idx.view(1, L, 1)).expand(batch_size, -1, -1)
        bwd = (idx.view(1, 1, L) > idx.view(1, L, 1)).expand(batch_size, -1, -1)

        if use_temporal_edges:
            temporal_mask = (dist_b > 0) & (dist_b <= window_size) & pair_valid
            if distance_decay:
                temp_w = torch.exp(-dist_b.float() / float(tau))
            else:
                temp_w = torch.ones_like(dist_b, dtype=torch.float32)
            m1 = temporal_mask & fwd
            rel_masks[:, 1] = m1
            rel_weights[:, 1] = temp_w * m1.float()
            m2 = temporal_mask & bwd
            rel_masks[:, 2] = m2
            rel_weights[:, 2] = temp_w * m2.float()

        if use_speaker_edges:
            same_speaker = speakers.unsqueeze(2).eq(speakers.unsqueeze(1))
            sp_mask = (dist_b > 0) & (dist_b <= window_size) & same_speaker & pair_valid
            if distance_decay:
                sp_w = torch.exp(-dist_b.float() / float(tau)) + 0.5
            else:
                sp_w = torch.full_like(dist_b, 1.5, dtype=torch.float32)
            rel_masks[:, 3] = sp_mask
            rel_weights[:, 3] = sp_w * sp_mask.float()

        # Mutual exclusion priority: self/temporal overrides speaker
        higher = rel_masks[:, 0] | rel_masks[:, 1] | rel_masks[:, 2]
        rel_masks[:, 3] = rel_masks[:, 3] & (~higher)
        rel_weights[:, 3] = rel_weights[:, 3] * rel_masks[:, 3].float()

        # KNN
        if use_knn and node_features is not None and knn_k and knn_k > 0:
            x = F.normalize(node_features, p=2, dim=-1)
            sim = torch.matmul(x, x.transpose(1, 2))  # (B,L,L)
            sim = sim.masked_fill(~pair_valid, -1e9)
            sim = sim.masked_fill(I == True, -1e9)
            k = int(max(1, min(knn_k, L - 1)))
            vals, idxs = torch.topk(sim, k=k, dim=-1)
            knn_mask = torch.zeros(batch_size, L, L, device=device, dtype=torch.bool)
            bidx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(idxs)
            ridx = torch.arange(L, device=device).view(1, -1, 1).expand_as(idxs)
            knn_mask[bidx, ridx, idxs] = True
            if knn_min_sim is not None:
                keep = (vals >= float(knn_min_sim))
                tmp = torch.zeros_like(knn_mask)
                tmp[bidx, ridx, idxs] = keep
                knn_mask = tmp
            # Always use mutual (bidirectional) KNN edges for reliability
            knn_mask = knn_mask & knn_mask.transpose(1, 2)
            higher_all = rel_masks[:, 0] | rel_masks[:, 1] | rel_masks[:, 2] | rel_masks[:, 3]
            knn_mask = knn_mask & (~higher_all)
            rel_masks[:, 4] = knn_mask
            sim_clamped = torch.clamp(sim, -1.0, 1.0)
            sim_norm = (sim_clamped + 1.0) / 2.0
            rel_weights[:, 4] = sim_norm * knn_mask.float()

        edge_weights_all = torch.zeros(batch_size, L, L, device=device)
        for r in range(R):
            edge_weights_all = torch.maximum(edge_weights_all, rel_weights[:, r])
        edge_weights_all = torch.maximum(edge_weights_all, edge_weights_all.transpose(1, 2))

        return ['self', 'temporal_fwd', 'temporal_bwd', 'same_speaker', 'knn'], rel_masks, rel_weights, edge_weights_all


class RelationalGATv2Layer(nn.Module):
    """Relation-aware GATv2 layer, using R-GCN basis decomposition for relation-specific projections."""

    def __init__(self, in_dim, out_dim, num_heads=8, num_relations=5, num_bases=4,
                 dropout=0.1, attn_dropout=None, use_edge_weights=True, use_bias=True, edge_drop=0.0):
        super(RelationalGATv2Layer, self).__init__()
        assert out_dim % num_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.num_relations = num_relations
        self.num_bases = max(1, num_bases)
        self.use_edge_weights = use_edge_weights
        self.edge_drop = float(edge_drop)

        # Basis parameters for Q/K/V
        self.basis_q = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.basis_k = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.basis_v = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.coeff_q = nn.Parameter(torch.Tensor(self.num_relations, self.num_bases))
        self.coeff_k = nn.Parameter(torch.Tensor(self.num_relations, self.num_bases))
        self.coeff_v = nn.Parameter(torch.Tensor(self.num_relations, self.num_bases))

        self.w_o = nn.Linear(out_dim, out_dim)
        # Support separate dropout rates for attention and output
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout if attn_dropout is not None else dropout)

        self.use_bias = use_bias
        if use_bias:
            self.attn_bias = nn.Parameter(torch.zeros(self.num_relations, self.num_heads, 1, 1))
        else:
            self.register_parameter('attn_bias', None)

        self._init_weights()

    def _init_weights(self):
        for t in [self.basis_q, self.basis_k, self.basis_v]:
            nn.init.xavier_uniform_(t)
        for t in [self.coeff_q, self.coeff_k, self.coeff_v]:
            nn.init.xavier_uniform_(t.unsqueeze(-1))
        nn.init.xavier_uniform_(self.w_o.weight)
        nn.init.zeros_(self.w_o.bias)

    def _compose(self, coeff, basis):
        # coeff: (R,B), basis: (B,in,out) -> (R,in,out)
        return torch.einsum('rb,bio->rio', coeff, basis)

    def forward(self, x, rel_masks, rel_weights=None, mask=None):
        B, L, _ = x.shape
        R = rel_masks.size(1)

        Wq = self._compose(self.coeff_q, self.basis_q)
        Wk = self._compose(self.coeff_k, self.basis_k)
        Wv = self._compose(self.coeff_v, self.basis_v)

        Qr, Kr, Vr = [], [], []
        for r in range(R):
            q = x @ Wq[r]
            k = x @ Wk[r]
            v = x @ Wv[r]
            Qr.append(q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3))
            Kr.append(k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3))
            Vr.append(v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3))

        scores_all = x.new_full((B, self.num_heads, L, L), -1e9)
        for r in range(R):
            if rel_masks[:, r].any():
                scores_r = torch.matmul(Qr[r], Kr[r].transpose(-2, -1)) / (self.head_dim ** 0.5)
                if self.use_bias and self.attn_bias is not None:
                    scores_r = scores_r + self.attn_bias[r]
                if mask is not None:
                    key_mask = mask.unsqueeze(1).unsqueeze(1)
                    scores_r = scores_r.masked_fill(key_mask == 0, -1e9)
                m = rel_masks[:, r].unsqueeze(1)
                if self.use_edge_weights and rel_weights is not None:
                    w = torch.clamp(rel_weights[:, r], min=1e-6)
                    scores_r = scores_r + torch.log(w).unsqueeze(1)
                scores_all = torch.where(m, scores_r, scores_all)

        if mask is not None:
            row_mask = mask.unsqueeze(1).unsqueeze(-1)
            scores_all = scores_all.masked_fill(row_mask == 0, -1e9)

        if self.training and self.edge_drop > 0:
            keep = torch.rand_like(scores_all, dtype=torch.float32) > self.edge_drop
            scores_all = torch.where(keep, scores_all, scores_all.new_full((), -1e9))

        attn = F.softmax(scores_all, dim=-1)
        attn = self.attn_dropout(attn)

        out = x.new_zeros(B, self.num_heads, L, self.head_dim)
        for r in range(R):
            if rel_masks[:, r].any():
                m = rel_masks[:, r].unsqueeze(1)
                out = out + torch.matmul(attn * m, Vr[r])

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.out_dim)
        out = self.w_o(out)
        out = self.dropout(out)
        return out


class GraphConversationEncoderRelational(nn.Module):
    """Relational conversation graph encoder: R-GAT + rebuild relational graph at each layer"""

    def __init__(self, hidden_dim, num_layers=2, num_heads=8, dropout=0.1,
                 attn_dropout=None, ffn_dropout=None,
                 window_size=3, use_speaker_edges=True, use_temporal_edges=True,
                 distance_decay=True, tau=2.0, use_residual=True,
                 rel_num_bases=4, rel_use_knn=True, rel_knn_k=6,
                 rel_knn_min_sim=0.5, rel_edge_drop=0.1):
        super(GraphConversationEncoderRelational, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.use_speaker_edges = use_speaker_edges
        self.use_temporal_edges = use_temporal_edges
        self.distance_decay = distance_decay
        self.tau = tau
        self.use_residual = use_residual

        self.rel_use_knn = rel_use_knn
        self.rel_knn_k = rel_knn_k
        self.rel_knn_min_sim = rel_knn_min_sim
        
        actual_attn_dropout = attn_dropout if attn_dropout is not None else dropout
        actual_ffn_dropout = ffn_dropout if ffn_dropout is not None else dropout

        self.gat_layers = nn.ModuleList([
            RelationalGATv2Layer(hidden_dim, hidden_dim, num_heads,
                                  num_relations=5, num_bases=rel_num_bases,
                                  dropout=dropout, attn_dropout=actual_attn_dropout,
                                  use_edge_weights=True, edge_drop=rel_edge_drop)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(actual_ffn_dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(actual_ffn_dropout)
            ) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.adjacency_builder = GraphAdjacencyBuilder()

    def forward(self, x, speakers, doc_lengths, mask=None):
        batch_size, max_doc_len, _ = x.shape

        output = x
        for i in range(self.num_layers):
            _, rel_masks, rel_weights, _ = self.adjacency_builder.build_relational_adjacency(
                speakers, doc_lengths, max_doc_len,
                window_size=self.window_size,
                use_speaker_edges=self.use_speaker_edges,
                use_temporal_edges=self.use_temporal_edges,
                distance_decay=self.distance_decay,
                tau=self.tau,
                node_features=output,
                use_knn=self.rel_use_knn,
                knn_k=self.rel_knn_k,
                knn_min_sim=self.rel_knn_min_sim,
            )
            gat_output = self.gat_layers[i](output, rel_masks, rel_weights, mask)

            if self.use_residual:
                output = self.layer_norms[i](output + gat_output)
            else:
                output = self.layer_norms[i](gat_output)

            ffn_output = self.ffns[i](output)
            if self.use_residual:
                output = self.ffn_norms[i](output + ffn_output)
            else:
                output = self.ffn_norms[i](ffn_output)

        return output

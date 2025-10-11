# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableFGWHead(nn.Module):
    """可微的二部FGW对齐头，返回pair级传输概率。

    row_norm: none | max | row_softmax（建议 row_softmax 提升召回）
    row_temp: row_softmax 的温度
    """

    def __init__(self, alpha=0.5, eps=0.1, max_iter=5, sinkhorn_iter=20, sinkhorn_eps=1e-6,
                 row_norm='row_softmax', row_temp=0.7):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter
        self.sinkhorn_iter = sinkhorn_iter
        self.sinkhorn_eps = sinkhorn_eps
        self.row_norm = row_norm
        self.row_temp = row_temp

    @staticmethod
    def _attribute_cost(E_feat, C_feat):
        E_norm = F.normalize(E_feat, p=2, dim=-1)
        C_norm = F.normalize(C_feat, p=2, dim=-1)
        sim = torch.matmul(E_norm, C_norm.t())
        cost = 1.0 - sim.clamp(-1.0, 1.0)
        return cost

    @staticmethod
    def _structure_cost(A_e, A_c, T):
        # 直接广播: (n_e, n_e, n_c, n_c)
        struct_diff = (A_e.unsqueeze(2).unsqueeze(3) - A_c.unsqueeze(0).unsqueeze(1)) ** 2
        # 求和 j,l 维度 -> (n_e, n_c)
        cost = torch.einsum('jl,ijkl->ik', T, struct_diff)
        return cost

    def _fgw_single(self, E_feat, C_feat, A_e, A_c, allowed_mask=None):
        device = E_feat.device
        n_e = E_feat.size(0)
        n_c = C_feat.size(0)

        if n_e == 0 or n_c == 0:
            return torch.zeros(n_e, n_c, device=device)

        mu_e = torch.full((n_e,), 1.0 / n_e, device=device)
        mu_c = torch.full((n_c,), 1.0 / n_c, device=device)

        # 属性代价
        attr_cost = self._attribute_cost(E_feat, C_feat)

        # 若提供先验mask：确保每行至少有一个可行列，避免退化
        if allowed_mask is not None:
            # 修补无可行列的行：选择属性代价最小的列
            row_any = allowed_mask.any(dim=1)
            if (~row_any).any():
                min_cols = torch.argmin(attr_cost[~row_any], dim=1)
                allowed_mask[~row_any] = False
                allowed_mask[~row_any, min_cols] = True
            # 修补无可行行的列（极少见）：选择属性代价最小的行
            col_any = allowed_mask.any(dim=0)
            if (~col_any).any():
                min_rows = torch.argmin(attr_cost[:, ~col_any], dim=0)
                allowed_mask[min_rows, ~col_any] = True

        # 初始化传输
        T = torch.outer(mu_e, mu_c)

        for _ in range(self.max_iter):
            # 结构代价
            struct_cost = self._structure_cost(A_e, A_c, T)
            total_cost = self.alpha * attr_cost + (1.0 - self.alpha) * struct_cost

            # 将不允许的(i,k)对代价抬高，抑制传输
            if allowed_mask is not None:
                total_cost = total_cost + (~allowed_mask) * 1e6

            # Sinkhorn 标准化
            K = torch.exp(-total_cost / self.eps).clamp_min(self.sinkhorn_eps)
            if allowed_mask is not None:
                K = K * allowed_mask.float()

            u = torch.ones_like(mu_e)
            v = torch.ones_like(mu_c)

            for _ in range(self.sinkhorn_iter):
                K_v = K @ v + self.sinkhorn_eps
                u = mu_e / K_v
                K_t_u = K.t() @ u + self.sinkhorn_eps
                v = mu_c / K_t_u

            T = (u.unsqueeze(1) * K) * v.unsqueeze(0)

        return T

    def forward(self, emotion_context, cause_context, edge_weights, pair_indices, doc_lengths,
                pred_future_cause=True, max_pair_distance=None):
        """
        Args:
            emotion_context: (B, L, D)
            cause_context:   (B, L, D)
            edge_weights:    (B, L, L) 图结构权重
            pair_indices:    (N_pairs, 3) [batch_idx, emo_idx, cause_idx]
            doc_lengths:     (B,)
            pred_future_cause: 是否允许未来原因（方向先验）
            max_pair_distance: 最大 |cause-emo|（距离先验），None 表示不额外限制

        Returns:
            pair_scores: (N_pairs,) 传输概率
            transports:  list[Tensor] 每个对话的传输矩阵
        """
        device = emotion_context.device
        num_pairs = pair_indices.size(0)
        pair_scores = torch.zeros(num_pairs, device=device)
        transports = {}

        if num_pairs == 0:
            return pair_scores, transports

        conv_to_rows = {}
        for row_i, conv_id in enumerate(pair_indices[:, 0].tolist()):
            conv_to_rows.setdefault(conv_id, []).append(row_i)

        for conv_id, rows in conv_to_rows.items():
            rows_t = torch.tensor(rows, dtype=torch.long, device=device)
            emo_idx = pair_indices[rows_t, 1]
            cause_idx = pair_indices[rows_t, 2]

            unique_emo, inv_e = torch.unique(emo_idx, sorted=True, return_inverse=True)
            unique_cau, inv_c = torch.unique(cause_idx, sorted=True, return_inverse=True)

            E_feat = emotion_context[conv_id, unique_emo]
            C_feat = cause_context[conv_id, unique_cau]

            A_full = edge_weights[conv_id]
            A_e = A_full.index_select(0, unique_emo).index_select(1, unique_emo)
            A_c = A_full.index_select(0, unique_cau).index_select(1, unique_cau)

            # 方向与距离先验：在 (unique_emo, unique_cau) 子网格上构造允许掩码
            e_pos = unique_emo.view(-1, 1).to(device)
            c_pos = unique_cau.view(1, -1).to(device)
            allowed = torch.ones(e_pos.size(0), c_pos.size(1), dtype=torch.bool, device=device)
            if not pred_future_cause:
                allowed = allowed & (c_pos <= e_pos)
            if max_pair_distance is not None:
                allowed = allowed & ((c_pos - e_pos).abs() <= int(max_pair_distance))

            T = self._fgw_single(E_feat, C_feat, A_e, A_c, allowed_mask=allowed)
            transports[int(conv_id)] = T

            # 行归一化/softmax 以放大相对差异
            if self.row_norm == 'row_softmax':
                logits = T / max(self.row_temp, 1e-6)
                T_norm = F.softmax(logits, dim=1)
            elif self.row_norm == 'max':
                row_max = T.max(dim=1, keepdim=True).values.clamp_min(1e-8)
                T_norm = T / row_max
            else:
                T_norm = T

            pair_scores[rows_t] = T_norm[inv_e, inv_c]

        return pair_scores, transports




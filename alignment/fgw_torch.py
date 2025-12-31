# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_ix_like(input, dim=0):
    """Create index tensor for gather/scatter operations"""
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def sparsemax(input, dim=-1):
   
    input = input.transpose(0, dim)
    original_size = input.size()
    input = input.reshape(input.size(0), -1)
    input = input.transpose(0, 1)
    dim = 1

    number_of_logits = input.size(dim)

    # Sort
    input_sorted, _ = torch.sort(input, dim=dim, descending=True)
    
    # Compute cumulative sum
    input_cumsum = input_sorted.cumsum(dim) - 1
    
    # Find threshold k
    rhos = _make_ix_like(input, dim)
    support = rhos * input_sorted > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)

    # Apply sparsemax
    output = torch.clamp(input - tau, min=0)

    # Restore shape
    output = output.transpose(1, 0)
    output = output.reshape(original_size)
    output = output.transpose(0, dim)

    return output


def entmax_bisect(input, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    
    if alpha == 1:
        return F.softmax(input, dim=dim)
    
    input = input.transpose(0, dim)
    original_size = input.size()
    input = input.reshape(input.size(0), -1)
    input = input.transpose(0, 1)
    dim = 1

    d = input.shape[dim]
    
    # Initialize tau search range
    input_sorted, _ = torch.sort(input, dim=dim, descending=True)
    
    tau_min = input_sorted[:, -1] - 1.0
    tau_max = input_sorted[:, 0] - (d ** (1.0 / (alpha - 1.0))) / d
    
    # Bisection search
    for _ in range(n_iter):
        tau = (tau_min + tau_max) / 2
        
        # Entmax transform: p = max(0, (x - tau) / (alpha - 1)) ^ (1 / (alpha - 1))
        p = torch.clamp(input - tau.unsqueeze(1), min=0)
        p = torch.pow(p, 1.0 / (alpha - 1.0))
        
        # Check normalization condition
        p_sum = p.sum(dim=dim)
        
        # Update search range
        tau_min = torch.where(p_sum > 1, tau, tau_min)
        tau_max = torch.where(p_sum <= 1, tau, tau_max)
    
    # Final output
    tau = (tau_min + tau_max) / 2
    p = torch.clamp(input - tau.unsqueeze(1), min=0)
    p = torch.pow(p, 1.0 / (alpha - 1.0))
    
    if ensure_sum_one:
        p = p / (p.sum(dim=dim, keepdim=True) + 1e-12)
    
    # Restore shape
    p = p.transpose(1, 0)
    p = p.reshape(original_size)
    p = p.transpose(0, dim)
    
    return p


def entmax15(input, dim=-1, n_iter=50):
    """Convenience function for Entmax-1.5 (recommended for attention mechanisms)"""
    return entmax_bisect(input, alpha=1.5, dim=dim, n_iter=n_iter)


class DifferentiableFGWHead(nn.Module):
    """Differentiable joint FGW alignment head: supports learnable Mahalanobis semantic cost + sparse normalization."""

    def __init__(self, alpha=0.5, eps=0.1, max_iter=5, sinkhorn_iter=20, sinkhorn_eps=1e-6,
                 row_norm='entmax', row_temp=0.7, entmax_alpha=1.5, entmax_bisect_iter=50,
                 use_uot=False, uot_rho=1.0,
                 attr_metric: str = 'maha', feat_dim: int | None = None):
        super().__init__()
        self.alpha = alpha  # FGW alpha (structure vs semantic)
        self.eps = eps
        self.max_iter = max_iter
        self.sinkhorn_iter = sinkhorn_iter
        self.sinkhorn_eps = sinkhorn_eps
        
        # Row normalization configuration
        self.row_norm = str(row_norm)
        if self.row_norm not in ['row_softmax', 'entmax', 'sparsemax', 'max', 'none']:
            print(f"Warning: Unknown row_norm '{self.row_norm}', falling back to 'entmax'")
            self.row_norm = 'entmax'
        
        self.row_temp = row_temp 
        self.entmax_alpha = float(entmax_alpha)  
        self.entmax_bisect_iter = int(entmax_bisect_iter)  
        
        self.use_uot = bool(use_uot)
        self.uot_rho = float(uot_rho)
        
        # Semantic cost configuration
        self.attr_metric = str(attr_metric)
        if self.attr_metric not in ['cos', 'maha']:
            self.attr_metric = 'maha'
        self.feat_dim = int(feat_dim) if feat_dim is not None else None
        if self.attr_metric == 'maha' and self.feat_dim is not None and self.feat_dim > 0:
            self.w_diag = nn.Parameter(torch.zeros(self.feat_dim))
        else:
            self.w_diag = None

    def _attribute_cost(self, E_feat, C_feat):
        if self.attr_metric == 'maha' and self.w_diag is not None:
            w = F.softplus(self.w_diag).to(E_feat.dtype).to(E_feat.device)
            eW = E_feat * w
            cW = C_feat * w
            eWe = (eW * E_feat).sum(dim=-1)
            cWc = (cW * C_feat).sum(dim=-1)
            eWc = eW @ C_feat.t()
            cost = eWe.unsqueeze(1) + cWc.unsqueeze(0) - 2.0 * eWc
            return torch.clamp(cost, min=0.0)
        else:
            E_norm = F.normalize(E_feat, p=2, dim=-1)
            C_norm = F.normalize(C_feat, p=2, dim=-1)
            sim = torch.matmul(E_norm, C_norm.t())
            cost = 1.0 - sim.clamp(-1.0, 1.0)
            return cost

    def _structure_cost(self, A_e, A_c, T):


        if A_e.numel() > 0:
            I_e = torch.eye(A_e.size(0), device=A_e.device, dtype=A_e.dtype)
            A_e = A_e * (1.0 - I_e)
        if A_c.numel() > 0:
            I_c = torch.eye(A_c.size(0), device=A_c.device, dtype=A_c.dtype)
            A_c = A_c * (1.0 - I_c)
        
        # Compute structure difference: (n_e, n_e, n_c, n_c)
        struct_diff = (A_e.unsqueeze(2).unsqueeze(3) - A_c.unsqueeze(0).unsqueeze(1)) ** 2
        # Sum over j,l dimensions -> (n_e, n_c)
        cost = torch.einsum('jl,ijkl->ik', T, struct_diff)
        return cost

    def _fgw_single(self, E_feat, C_feat, A_e, A_c, allowed_mask=None, row_cap: int | None = None, col_cap: int | None = None):
        device = E_feat.device
        n_e = E_feat.size(0)
        n_c = C_feat.size(0)

        if n_e == 0 or n_c == 0:
            return torch.zeros(n_e, n_c, device=device)

        mu_e = torch.full((n_e,), 1.0, device=device) if (row_cap is None or row_cap <= 0) else torch.full((n_e,), float(row_cap), device=device)
        mu_c = torch.full((n_c,), 1.0, device=device) if (col_cap is None or col_cap <= 0) else torch.full((n_c,), float(col_cap), device=device)

        # Attribute cost
        attr_cost = self._attribute_cost(E_feat, C_feat)

        if allowed_mask is not None:
            # Fix rows with no feasible columns: select column with minimum attribute cost
            row_any = allowed_mask.any(dim=1)
            if (~row_any).any():
                min_cols = torch.argmin(attr_cost[~row_any], dim=1)
                allowed_mask[~row_any] = False
                allowed_mask[~row_any, min_cols] = True
            # Fix columns with no feasible rows (rare): select row with minimum attribute cost
            col_any = allowed_mask.any(dim=0)
            if (~col_any).any():
                min_rows = torch.argmin(attr_cost[:, ~col_any], dim=0)
                allowed_mask[min_rows, ~col_any] = True

        # Initialize transport
        T = torch.outer(mu_e, mu_c)
        T = T / (T.sum() + self.sinkhorn_eps)

        for _ in range(self.max_iter):
            # Structure cost
            struct_cost = self._structure_cost(A_e, A_c, T)
            total_cost = self.alpha * attr_cost + (1.0 - self.alpha) * struct_cost

            # Increase cost for disallowed (i,k) pairs to suppress transport
            if allowed_mask is not None:
                total_cost = total_cost + (~allowed_mask) * 1e6

            # Sinkhorn normalization
            K = torch.exp(-total_cost / self.eps).clamp_min(self.sinkhorn_eps)
            if allowed_mask is not None:
                K = K * allowed_mask.float()

            u = torch.ones_like(mu_e)
            v = torch.ones_like(mu_c)

            if self.use_uot:
                # Unbalanced OT (Chizat et al., 2018)
                tau = float(self.uot_rho) / float(self.uot_rho + self.eps)
                tau = max(1e-6, min(1.0, tau))
                for _ in range(self.sinkhorn_iter):
                    Kv = K @ v + self.sinkhorn_eps
                    u = (mu_e / Kv).clamp_min(self.sinkhorn_eps) ** tau
                    Ktu = K.t() @ u + self.sinkhorn_eps
                    v = (mu_c / Ktu).clamp_min(self.sinkhorn_eps) ** tau
            else:
                for _ in range(self.sinkhorn_iter):
                    K_v = K @ v + self.sinkhorn_eps
                    u = mu_e / K_v
                    K_t_u = K.t() @ u + self.sinkhorn_eps
                    v = mu_c / K_t_u

            T = (u.unsqueeze(1) * K) * v.unsqueeze(0)

        return T

    def forward(
        self,
        emotion_context,
        cause_context,
        edge_weights_e,
        edge_weights_c,
        pair_indices,
        doc_lengths,
        pred_future_cause=False,
        max_pair_distance=None,
        row_cap: int | None = None,
        col_cap: int | None = None,
    ):
                
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

            A_full_e = edge_weights_e[conv_id]
            A_full_c = edge_weights_c[conv_id]
            A_e = A_full_e.index_select(0, unique_emo).index_select(1, unique_emo)
            A_c = A_full_c.index_select(0, unique_cau).index_select(1, unique_cau)

            # Direction and distance prior: construct allowed mask on (unique_emo, unique_cau) sub-grid
            e_pos = unique_emo.view(-1, 1).to(device)
            c_pos = unique_cau.view(1, -1).to(device)
            allowed = torch.ones(e_pos.size(0), c_pos.size(1), dtype=torch.bool, device=device)
            if not pred_future_cause:
                allowed = allowed & (c_pos <= e_pos)
            if max_pair_distance is not None:
                allowed = allowed & ((c_pos - e_pos).abs() <= int(max_pair_distance))

            T = self._fgw_single(E_feat, C_feat, A_e, A_c, allowed_mask=allowed, row_cap=row_cap, col_cap=col_cap)
            transports[int(conv_id)] = T

            # Row normalization: supports multiple sparsification strategies
            if self.row_norm == 'row_softmax':
                # Standard softmax (all non-zero, over-smooth)
                logits = T / max(self.row_temp, 1e-6)
                T_norm = F.softmax(logits, dim=1)
            
            elif self.row_norm == 'entmax':
                # Entmax-α (recommended): balances sparsity and smoothness
                logits = T / max(self.row_temp, 1e-6)
                T_norm = entmax_bisect(
                    logits, 
                    alpha=self.entmax_alpha,
                    dim=1, 
                    n_iter=self.entmax_bisect_iter
                )
            
            elif self.row_norm == 'sparsemax':
                # Sparsemax: true sparse (some positions are 0)
                logits = T / max(self.row_temp, 1e-6)
                T_norm = sparsemax(logits, dim=1)
            
            elif self.row_norm == 'max':
                # Max normalization
                row_max = T.max(dim=1, keepdim=True).values.clamp_min(1e-8)
                T_norm = T / row_max
            
            else:  # 'none'
                # No normalization
                T_norm = T

            pair_scores[rows_t] = T_norm[inv_e, inv_c]

        return pair_scores, transports
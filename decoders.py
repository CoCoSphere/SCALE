# -*- coding: utf-8 -*-

"""
Global bipartite b-matching decoder (Min-Cost Flow, SSP + potentials).
Usage: Merge candidate edges (emotion_idx=i, cause_idx=j) with their scores for each conversation,
perform global pair selection under row/col capacity and global "threshold price" constraints, 
allowing empty rows (no forced minimum of 1 per row).

Entry: global_mcmf_decode(pair_indices, probs, conv_ids, ...)
Returns: preds tensor (same length as pair_indices) 0/1.
"""


from typing import List, Dict, Tuple
import math

import torch


class _Edge:
    __slots__ = ("to", "rev", "cap", "cost")

    def __init__(self, to: int, rev: int, cap: int, cost: float):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost


class _MCMF:
    def __init__(self, n: int):
        self.n = n
        self.g: List[List[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: float):
        a = _Edge(v, len(self.g[v]), cap, cost)
        b = _Edge(u, len(self.g[u]), 0, -cost)
        self.g[u].append(a)
        self.g[v].append(b)

    def min_cost_flow_ssp(self, s: int, t: int, max_flow: int) -> Tuple[int, float]:
        import heapq

        n = self.n
        flow = 0
        cost = 0.0
        pot = [0.0] * n  # potentials

        while flow < max_flow:
            dist = [float("inf")] * n
            dist[s] = 0.0
            prev_v = [-1] * n
            prev_e = [-1] * n

            hq = [(0.0, s)]
            while hq:
                d, v = heapq.heappop(hq)
                if d > dist[v]:
                    continue
                for ei, e in enumerate(self.g[v]):
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + pot[v] - pot[e.to]
                    if nd < dist[e.to]:
                        dist[e.to] = nd
                        prev_v[e.to] = v
                        prev_e[e.to] = ei
                        heapq.heappush(hq, (nd, e.to))

            if dist[t] == float("inf"):
                # No augmenting path
                break

            for i in range(n):
                if dist[i] < float("inf"):
                    pot[i] += dist[i]

            # In this implementation, path min residual is often 1 (E->C capacity 1), but still compute residual for generality.
            addf = max_flow - flow
            v = t
            while v != s:
                e = self.g[prev_v[v]][prev_e[v]]
                addf = min(addf, e.cap)
                v = prev_v[v]

            v = t
            while v != s:
                e = self.g[prev_v[v]][prev_e[v]]
                re = self.g[v][e.rev]
                e.cap -= addf
                re.cap += addf
                v = prev_v[v]

            flow += addf
            cost += addf * pot[t]

        return flow, cost


def _to_logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p) - math.log(1.0 - p)


def _build_and_solve_mcmf_ortools(
    num_left: int,
    num_right: int,
    edges: List[Tuple[int, int, float]],
    row_cap: int = 1,
    col_cap: int = 1,
    lambda_cost: float = 0.5,
    score_space: str = "prob",
    eps: float = 1e-6,
    cost_scale: int = 1000,
) -> List[Tuple[int, int]]:

    import ctypes
    import os
    import ortools
        
        # Use correct library filename
    libpb_path = os.path.join(os.path.dirname(ortools.__file__), '.libs/libprotobuf.so.29.3.0')
    if os.path.exists(libpb_path):
            ctypes.CDLL(libpb_path, mode=ctypes.RTLD_GLOBAL)
    else:
            # If not exists, try to find other possible protobuf libraries
            lib_dir = os.path.join(os.path.dirname(ortools.__file__), '.libs')
            for file in os.listdir(lib_dir):
                if file.startswith('libprotobuf.so'):
                    full_path = os.path.join(lib_dir, file)
                    ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                    break
    try:
        from ortools.graph.python import min_cost_flow as _mcf_mod
    except Exception as e:
        raise ImportError("OR-Tools not available") from e

    N = num_left + num_right + 2
    s = num_left + num_right
    t = s + 1
    mcf = _mcf_mod.SimpleMinCostFlow()

    # Adapt to new/old API method names
    def add_arc(u: int, v: int, cap: int, cost: int):
        if hasattr(mcf, 'add_arc_with_capacity_and_unit_cost'):
            mcf.add_arc_with_capacity_and_unit_cost(u, v, int(max(0, cap)), int(cost))
        else:
            mcf.AddArcWithCapacityAndUnitCost(u, v, int(max(0, cap)), int(cost))
    def set_supply(node: int, supply: int):
        if hasattr(mcf, 'set_node_supply'):
            mcf.set_node_supply(node, int(supply))
        else:
            mcf.SetNodeSupply(node, int(supply))
    # s -> left
    for u in range(num_left):
        add_arc(s, u, row_cap, 0)

    # left -> right
    if score_space == "logit":
        lam = _to_logit(lambda_cost, eps)
        for (u, v, sc) in edges:
            c = lam - _to_logit(float(sc), eps)
            c_int = int(round(c * cost_scale))
            add_arc(u, num_left + v, 1, c_int)
    else:
        lam = float(lambda_cost)
        for (u, v, sc) in edges:
            c = lam - float(sc)
            c_int = int(round(c * cost_scale))
            add_arc(u, num_left + v, 1, c_int)

    # right -> t
    for v in range(num_right):
        add_arc(num_left + v, t, col_cap, 0)

    # left -> t (rejection edges)
    for u in range(num_left):
        add_arc(u, t, row_cap, 0)

    # supplies
    total_supply = int(num_left * row_cap)
    for i in range(N):
        if i == s:
            set_supply(i, total_supply)
        elif i == t:
            set_supply(i, -total_supply)
        else:
            set_supply(i, 0)
    # Compatible with new/old API: solve() or Solve()
    if hasattr(mcf, 'solve'):
        status = mcf.solve()
    else:
        status = mcf.Solve()
    # Compatible with new/old API return status
    ok = (status == 0) or (status is True)
    if not ok:
        try:
            from ortools.graph.python import min_cost_flow as _new
            if hasattr(_new, "Status") and hasattr(_new.Status, "OPTIMAL"):
                ok = ok or (status == _new.Status.OPTIMAL)
        except Exception:
            pass
    if not ok:
        try:
            ok = "OPTIMAL" in str(status)
        except Exception:
            ok = False
    if not ok:
        raise RuntimeError(f"OR-Tools MinCostFlow failed with status={status}")

    # Read solution: compatible with new/old API for num_arcs/tail/head/flow
    if hasattr(mcf, 'NumArcs'):
        _num = mcf.NumArcs()
        _tail = mcf.Tail
        _head = mcf.Head
        _flow = mcf.Flow
    else:
        _num = mcf.num_arcs()
        _tail = mcf.tail
        _head = mcf.head
        _flow = mcf.flow

    selected: List[Tuple[int, int]] = []
    for a in range(_num):
        u = _tail(a)
        v = _head(a)
        if 0 <= u < num_left and num_left <= v < num_left + num_right:
            if _flow(a) >= 1:
                selected.append((u, v - num_left))
    return selected

def _build_and_solve_mcmf(
    num_left: int,
    num_right: int,
    edges: List[Tuple[int, int, float]],
    row_cap: int = 1,
    col_cap: int = 1,
    lambda_cost: float = 0.5,
    score_space: str = "prob",
    eps: float = 1e-6,
) -> List[Tuple[int, int]]:

    N = num_left + num_right + 2
    s = num_left + num_right
    t = s + 1
    mcmf = _MCMF(N)

    # s -> left
    for u in range(num_left):
        mcmf.add_edge(s, u, row_cap, 0.0)

    # right -> t
    for v in range(num_right):
        mcmf.add_edge(num_left + v, t, col_cap, 0.0)

    # left -> right (candidate edges)
    for u, v, sc in edges:
        if score_space == "logit":
            scv = _to_logit(sc, eps)
            # Put threshold in logit space as well
            lam = _to_logit(lambda_cost, eps)
            cost = lam - scv
        else:
            lam = lambda_cost
            cost = lam - float(sc)
        mcmf.add_edge(u, num_left + v, 1, cost)

    # left -> t rejection edges
    for u in range(num_left):
        mcmf.add_edge(u, t, row_cap, 0.0)

    max_flow = row_cap * num_left
    mcmf.min_cost_flow_ssp(s, t, max_flow)

    selected: List[Tuple[int, int]] = []
    # Traverse left node out-edges, find E->C edges with flow (residual decrease means selected)
    for u in range(num_left):
        for e in mcmf.g[u]:
            # Points to right nodes and is candidate edge: to ∈ [num_left, num_left + num_right)
            if num_left <= e.to < num_left + num_right:
                used = (e.cap == 0)  # Initial cap=1, exhausted means cap=0
                if used:
                    v = e.to - num_left
                    selected.append((u, v))
    return selected


@torch.no_grad()
def global_mcmf_decode(
    pair_indices: torch.Tensor,  # (N,3) [b, e, c]
    probs: torch.Tensor,         # (N,)
    conv_ids: List,              # len=N, conversation IDs (aligned row-wise with pair_indices)
    row_cap: int = 1,
    col_cap: int = 1,
    lambda_cost: float = 0.5,
    score_space: str = "prob",
    eps: float = 1e-6,
    pre_topk_per_row: int | None = None,
    pre_min_prob: float | None = None,
    pre_min_logit: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Perform global MCMF decoding for multiple conversations in a batch, return row-wise 0/1 predictions."""
    assert pair_indices.dim() == 2 and pair_indices.size(1) == 3
    assert probs.dim() == 1 and probs.numel() == pair_indices.size(0)

    N = pair_indices.size(0)
    preds = torch.zeros(N, dtype=torch.long, device=device)

    # Grouping: aggregate row indices by conversation
    conv_to_rows: Dict = {}
    for idx, cid in enumerate(conv_ids):
        conv_to_rows.setdefault(cid, []).append(idx)

    for cid, rows in conv_to_rows.items():
        if not rows:
            continue
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        sub = pair_indices[rows_t]  # (R,3)
        sub_probs = probs[rows_t]

        # Map to local left/right numbering
        emos = sub[:, 1].tolist()
        caus = sub[:, 2].tolist()
        uniq_e = sorted(set(emos))
        uniq_c = sorted(set(caus))
        map_e = {e: i for i, e in enumerate(uniq_e)}
        map_c = {c: i for i, c in enumerate(uniq_c)}

        # Optional pre-filtering: keep top-K per left, with score >= threshold
        per_left: Dict[int, List[Tuple[int, float, int]]] = {}
        for r, (b, e0, c0) in enumerate(sub.tolist()):
            u = map_e[e0]
            v = map_c[c0]
            sc_prob = float(sub_probs[r].item())

            # Dual-domain threshold and automatic threshold
            keep = True
            if score_space == "prob":
                thr_prob = None
                if pre_min_prob is not None:
                    thr_prob = float(pre_min_prob)
                elif pre_min_logit is not None:
                    thr_prob = 1.0 / (1.0 + math.exp(-float(pre_min_logit)))
                if thr_prob is not None and sc_prob < thr_prob:
                    keep = False
            else:  # 'logit'
                thr_logit = None
                if pre_min_logit is not None:
                    thr_logit = float(pre_min_logit)
                elif pre_min_prob is not None:
                    thr_logit = _to_logit(float(pre_min_prob), eps)
                if thr_logit is not None:
                    sc_logit = _to_logit(sc_prob, eps)
                    if sc_logit < thr_logit:
                        keep = False
            if not keep:
                continue
            per_left.setdefault(u, []).append((v, sc_prob, rows[r]))

        edges: List[Tuple[int, int, float]] = []
        row_of_pair: Dict[Tuple[int, int], int] = {}
        for u, lst in per_left.items():
            if pre_topk_per_row is not None and pre_topk_per_row > 0:
                lst = sorted(lst, key=lambda x: x[1], reverse=True)[:int(pre_topk_per_row)]
            for v, sc, rid in lst:
                edges.append((u, v, sc))
                row_of_pair.setdefault((u, v), rid)

        sel = _build_and_solve_mcmf_ortools(
            num_left=len(uniq_e),
            num_right=len(uniq_c),
            edges=edges,
            row_cap=int(row_cap),
            col_cap=int(col_cap),
            lambda_cost=float(lambda_cost),
            score_space=str(score_space),
            eps=float(eps),
        )

        for (u, v) in sel:
            ridx = row_of_pair.get((u, v), None)
            if ridx is not None:
                preds[ridx] = 1

    return preds
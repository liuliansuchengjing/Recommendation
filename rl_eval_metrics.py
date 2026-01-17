
# -*- coding: utf-8 -*-
"""
rl_eval_metrics.py

Evaluation helpers for your "Base candidate TopK + RL re-ranking/selection" pipeline.

What it computes (fair, comparable):
1) Step-wise next-item ranking metrics (teacher-forcing in the sense of using the *current generated prefix*):
   - Hit@K (=Recall@K for single ground-truth next item)
   - Precision@K (=Hit@K / K for single ground-truth)
   - F1@K
   - MAP@K (single relevant -> 1/rank if hit else 0)
   - NDCG@K (single relevant -> 1/log2(rank+2) if hit else 0)
   for BOTH:
     a) Base model ranking (topK from base probs)
     b) RL policy ranking (topK from policy logits over candidate set)

2) Path metrics for fixed horizon m (open-loop rollout):
   - token Acc@m (position-wise accuracy in the window)
   - Exact@m (all m positions match)
   - Overlap@m (multiset overlap / m; equals P=R=F1 when |GT|=|Pred|=m)
   - MAP@m, NDCG@m treating the GT window as relevant set (binary relevance)
   for BOTH base-greedy rollout and RL-greedy rollout.

Notes:
- This file assumes PAD token id is 0.
- RLPathOptimizer / Env are from rl_adjuster_new.py.
"""

from typing import Dict, List, Tuple, Optional
import math
import torch


def _single_relevant_rank_metrics(topk_ids: torch.Tensor, gold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    topk_ids: [B,K] (global item ids)
    gold:     [B]   (global item ids)
    returns: hit [B] (0/1), ap [B], ndcg [B]
    """
    B, K = topk_ids.shape
    # positions: 0..K-1 where topk matches gold
    match = (topk_ids == gold.view(-1, 1))
    hit = match.any(dim=1).float()
    # rank (1-indexed) if hit else large
    # take first True index
    idx = torch.where(match, torch.arange(K, device=topk_ids.device).view(1, -1).expand(B, -1), torch.full((B, K), K, device=topk_ids.device))
    first = idx.min(dim=1).values  # 0..K-1 or K if no hit
    rank = first + 1  # 1..K or K+1 (no hit)
    ap = torch.where(hit > 0, 1.0 / rank.float(), torch.zeros_like(hit))
    ndcg = torch.where(hit > 0, 1.0 / torch.log2(rank.float() + 1.0), torch.zeros_like(hit))
    return hit, ap, ndcg


def _update_agg(agg: Dict[str, float], key: str, val: torch.Tensor):
    agg[key] = agg.get(key, 0.0) + float(val.sum().detach().cpu())


def _finalize_step_metrics(agg: Dict[str, float], denom: float, K: int, prefix: str) -> Dict[str, float]:
    """
    agg contains sums over valid steps/samples:
      - {prefix}_hit_sum
      - {prefix}_ap_sum
      - {prefix}_ndcg_sum
    denom is number of evaluated (sample,step) pairs.
    """
    if denom <= 0:
        return {f"{prefix}hits@{K}": 0.0, f"{prefix}map@{K}": 0.0, f"{prefix}NDCG@{K}": 0.0,
                f"{prefix}precision@{K}": 0.0, f"{prefix}recall@{K}": 0.0, f"{prefix}f1@{K}": 0.0}

    hit = agg.get(f"{prefix}_hit_sum", 0.0) / denom
    mp = agg.get(f"{prefix}_ap_sum", 0.0) / denom
    nd = agg.get(f"{prefix}_ndcg_sum", 0.0) / denom
    prec = hit / float(K)
    rec = hit  # single relevant
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)

    return {
        f"{prefix}hits@{K}": hit,
        f"{prefix}map@{K}": mp,
        f"{prefix}NDCG@{K}": nd,
        f"{prefix}precision@{K}": prec,
        f"{prefix}recall@{K}": rec,
        f"{prefix}f1@{K}": f1,
    }


def _window_overlap_stats(pred_win: torch.Tensor, gt_win: torch.Tensor) -> float:
    """
    pred_win, gt_win: [m] int tensors
    multiset overlap / m
    """
    from collections import Counter
    p = Counter(pred_win.tolist())
    g = Counter(gt_win.tolist())
    tp = sum(min(p[x], g.get(x, 0)) for x in p)
    return tp / max(1, len(gt_win))


def _window_map_ndcg(pred_win: torch.Tensor, gt_set: set) -> Tuple[float, float]:
    """
    Treat gt_set as relevant (binary). pred_win is an ordered list length m.
    MAP here: mean of precisions at each hit position divided by number of relevant hits in window (AP on binary).
    NDCG: DCG with log discount / IDCG.
    """
    m = len(pred_win)
    hits = []
    for i, it in enumerate(pred_win.tolist()):
        if it in gt_set:
            hits.append(i)
    if len(hits) == 0:
        return 0.0, 0.0

    # AP
    precs = []
    hit_cnt = 0
    for i in range(m):
        if pred_win[i].item() in gt_set:
            hit_cnt += 1
            precs.append(hit_cnt / float(i + 1))
    ap = sum(precs) / float(len(gt_set))  # normalize by |GT| like your earlier code

    # NDCG
    dcg = 0.0
    for i in range(m):
        rel = 1.0 if pred_win[i].item() in gt_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    # ideal: all relevant items first (up to m)
    ideal_rels = [1.0] * min(len(gt_set), m) + [0.0] * (m - min(len(gt_set), m))
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        if rel > 0:
            idcg += rel / math.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ap, ndcg


@torch.no_grad()
def rollout_episode_greedy(
    rl,
    tgt: torch.Tensor,
    tgt_timestamp: torch.Tensor,
    tgt_idx: torch.Tensor,
    ans: torch.Tensor,
    graph=None,
    hypergraph_list=None,
    mode: str = "rl",  # "rl" or "base"
) -> Dict[str, torch.Tensor]:
    """
    Roll out a full episode (until valid_len) using:
      - mode="rl": greedy action = argmax(policy_logits) over candidate set
      - mode="base": greedy action = argmax(base_probs) (mapped into candidate index)
    Returns:
      gen_seq: [B,L]
      valid_lens: [B]
      seed_len: int
      per_step_base_topk: List[Tensor[B,topk]] (length=max_steps)
      per_step_policy_topk: List[Tensor[B,topk]] (length=max_steps; only if mode="rl", else None)
    """
    env = rl.env
    # state = env.reset(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
    start_t = getattr(env, "min_start", 0)
    state = env.reset(
        tgt, tgt_timestamp, tgt_idx, ans,
        start_t=start_t,
        graph=graph,
        hypergraph_list=hypergraph_list
    )

    rl._lazy_init(int(state["candidate_features"].size(-1)))
    rl.policy.eval()

    B, L = env.gen_seq.shape
    seed_len = getattr(env, "seed_len", 2)
    max_steps = env.max_steps

    per_step_base_topk = []
    per_step_policy_topk = []

    for _ in range(max_steps):
        cand_feat = state["candidate_features"]         # [B,Kc,F]
        cand_ids = state["candidate_ids"]               # [B,Kc]
        base_probs = env._pre_base_probs                # [B,V]

        # base ranking topK over full item space
        K_eval = int(env.topk)
        base_topk = torch.topk(base_probs, k=min(K_eval, base_probs.size(-1)), dim=-1).indices
        per_step_base_topk.append(base_topk)

        logits, _ = rl.policy(cand_feat)                # [B,Kc]
        # policy ranking topK over candidate set (mapped to global ids)
        pol_topk_idx = torch.topk(logits, k=min(K_eval, logits.size(-1)), dim=-1).indices
        pol_topk = cand_ids.gather(1, pol_topk_idx)
        per_step_policy_topk.append(pol_topk)

        if mode == "rl":
            action_idx = logits.argmax(dim=-1)  # [B]
        elif mode == "base":
            # choose base argmax (global), map to candidate index
            chosen_global = base_probs.argmax(dim=-1)  # [B]
            # find index within cand_ids; fallback to 0
            match = (cand_ids == chosen_global.view(-1, 1))
            found = match.any(dim=1)
            # argmax on boolean gives first True if exists else 0
            cand_pos = match.float().argmax(dim=1)
            action_idx = torch.where(found, cand_pos, torch.zeros_like(cand_pos))
        else:
            raise ValueError(f"Unknown mode={mode}")

        state, _, _, _ = env.step_env(action_idx)

    return {
        "gen_seq": env.gen_seq.detach().clone(),
        "valid_lens": env.valid_lens.detach().clone(),
        "seed_len": torch.tensor(seed_len, device=env.device),
        "base_topk_steps": per_step_base_topk,
        "policy_topk_steps": per_step_policy_topk,
    }


@torch.no_grad()
def evaluate_policy_with_ranking_metrics(
    rl,
    data_loader,
    graph,
    hypergraph_list,
    device: torch.device,
    max_batches: int = 20,
    k_list: Optional[List[int]] = None,
    m_list: Optional[List[int]] = None,
    schemeA_skip_short: bool = True,
) -> Dict[str, float]:
    """
    Runs evaluation and returns a flat dict of metrics.

    It computes:
    - Step-wise ranking metrics for base & policy at K in k_list (Hit/MAP/NDCG/P/R/F1).
    - Path metrics for base-greedy rollout & RL-greedy rollout at horizon m in m_list (Acc/Exact/Overlap/MAP/NDCG).
    """
    if k_list is None:
        k_list = [1, 3, 5, 10]
    if m_list is None:
        m_list = [3, 5, 7, 9]

    rl.policy.eval()

    # aggregate containers
    step_aggs = {("base", K): {"base_hit_sum": 0.0, "base_ap_sum": 0.0, "base_ndcg_sum": 0.0, "denom": 0.0}
                 for K in k_list}
    step_aggs.update({("policy", K): {"policy_hit_sum": 0.0, "policy_ap_sum": 0.0, "policy_ndcg_sum": 0.0, "denom": 0.0}
                      for K in k_list})

    path_aggs = {("base", m): {"acc_sum": 0.0, "exact_sum": 0.0, "overlap_sum": 0.0, "map_sum": 0.0, "ndcg_sum": 0.0, "windows": 0.0}
                 for m in m_list}
    path_aggs.update({("policy", m): {"acc_sum": 0.0, "exact_sum": 0.0, "overlap_sum": 0.0, "map_sum": 0.0, "ndcg_sum": 0.0, "windows": 0.0}
                      for m in m_list})

    for bi, batch in enumerate(data_loader):
        if bi >= max_batches:
            break
        tgt, tgt_timestamp, tgt_idx, ans = batch[0], batch[1], batch[2], batch[3]
        tgt = tgt.to(device); tgt_timestamp = tgt_timestamp.to(device); tgt_idx = tgt_idx.to(device); ans = ans.to(device)

        # Ensure policy exists before loading or evaluating
        rl.ensure_initialized(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)

        # 1) Base rollout (greedy base argmax) for path metrics + step ranking (base)
        out_base = rollout_episode_greedy(
            rl, tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list, mode="base"
        )
        # 2) Policy rollout (greedy policy argmax) for path metrics + step ranking (policy & base)
        out_pol = rollout_episode_greedy(
            rl, tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list, mode="rl"
        )

        # ---------- step-wise ranking metrics ----------
        # For each step s, gold next item is at position next_pos = step+1.
        # In env, first action writes position 2 (seed_len=2), so we evaluate from next_pos=2 onward.
        for K in k_list:
            # base step metrics from out_pol's base_topk_steps (base ranking on the *policy-generated prefix*)
            # and policy step metrics from out_pol's policy_topk_steps.
            # This makes both evaluated on the same evolving state distribution (policy rollout).
            denom = 0.0
            base_hit_sum = base_ap_sum = base_ndcg_sum = 0.0
            pol_hit_sum = pol_ap_sum = pol_ndcg_sum = 0.0

            base_steps = out_pol["base_topk_steps"]
            pol_steps = out_pol["policy_topk_steps"]
            gen_seq = out_pol["gen_seq"]
            valid_lens = out_pol["valid_lens"]
            seed_len = int(out_pol["seed_len"].item())

            for s in range(len(base_steps)):
                next_pos = (seed_len - 1) + 1 + s  # step starts at seed_len-1, next_pos = step+1
                # Actually env.step starts at 1 when seed_len=2, and increases by 1 each loop.
                # With seed_len=2 -> first next_pos=2.
                next_pos = 2 + s

                if next_pos >= gen_seq.size(1):
                    break
                gold = tgt[:, next_pos]  # compare to ORIGINAL gold next item at this position

                # valid mask: only samples with enough length
                mask = (valid_lens > next_pos) & (gold != 0)
                if mask.sum().item() == 0:
                    continue

                base_topk = base_steps[s][:, :K]
                pol_topk = pol_steps[s][:, :K]

                hit_b, ap_b, nd_b = _single_relevant_rank_metrics(base_topk, gold)
                hit_p, ap_p, nd_p = _single_relevant_rank_metrics(pol_topk, gold)

                hit_b = hit_b[mask]; ap_b = ap_b[mask]; nd_b = nd_b[mask]
                hit_p = hit_p[mask]; ap_p = ap_p[mask]; nd_p = nd_p[mask]

                denom += float(mask.sum().item())
                base_hit_sum += float(hit_b.sum().item())
                base_ap_sum += float(ap_b.sum().item())
                base_ndcg_sum += float(nd_b.sum().item())

                pol_hit_sum += float(hit_p.sum().item())
                pol_ap_sum += float(ap_p.sum().item())
                pol_ndcg_sum += float(nd_p.sum().item())

            # accumulate
            step_aggs[("base", K)]["base_hit_sum"] += base_hit_sum
            step_aggs[("base", K)]["base_ap_sum"] += base_ap_sum
            step_aggs[("base", K)]["base_ndcg_sum"] += base_ndcg_sum
            step_aggs[("base", K)]["denom"] += denom

            step_aggs[("policy", K)]["policy_hit_sum"] += pol_hit_sum
            step_aggs[("policy", K)]["policy_ap_sum"] += pol_ap_sum
            step_aggs[("policy", K)]["policy_ndcg_sum"] += pol_ndcg_sum
            step_aggs[("policy", K)]["denom"] += denom

        # ---------- path metrics (open-loop fixed horizon m, Scheme A via skip) ----------
        # We compute sliding windows on the generated sequences vs gold.
        for tag, out in [("base", out_base), ("policy", out_pol)]:
            gen_seq = out["gen_seq"]
            valid_lens = out["valid_lens"]
            B, L = gen_seq.shape

            seed_len = int(out.get("seed_len", torch.tensor(2, device=device)).item())
            for m in m_list:
                acc_sum = exact_sum = overlap_sum = map_sum = ndcg_sum = 0.0
                windows = 0.0

                for b in range(B):
                    vlen = int(valid_lens[b].item())
                    # start positions t where we compare next m items: positions t+1..t+m
                    # ensure these predicted positions exist (>=2) and within vlen
                                        # --- OLD (kept for reference): sliding windows over full valid length ---
                    # for tpos in range(1, vlen - m):
                    #                         start = tpos
                    #                         s = start + 1
                    #                         e = start + 1 + m
                    #                         if e > vlen:
                    #                             continue
                    #                         gt_win = tgt[b, s:e]
                    #                         pred_win = gen_seq[b, s:e]
                    #                         # Scheme A: if GT has PAD (shouldn't within vlen) or pred has PAD, skip or keep.
                    #                         if schemeA_skip_short:
                    #                             if (gt_win == 0).any():
                    #                                 continue
                    #                         windows += 1.0
                    #
                    #                         # token accuracy
                    #                         acc = (pred_win == gt_win).float().mean().item()
                    #                         acc_sum += acc
                    #                         # exact
                    #                         exact_sum += 1.0 if (pred_win == gt_win).all().item() else 0.0
                    #                         # overlap
                    #                         overlap_sum += _window_overlap_stats(pred_win, gt_win)
                    #                         # map/ndcg treating GT window as relevant set
                    #                         gt_set = set(gt_win.tolist())
                    #                         ap, nd = _window_map_ndcg(pred_win, gt_set)
                    #                         map_sum += ap
                    #                         ndcg_sum += nd
                    # --- NEW: evaluate only within the generated open-loop segment (path itself) ---
                    pred_len = int((gen_seq[b, seed_len:] != 0).sum().item())
                    if pred_len < m:
                        if schemeA_skip_short:
                            continue
                        continue
                    start_min = seed_len - 1
                    start_max_excl = seed_len + pred_len - m  # exclusive
                    for start in range(start_min, start_max_excl):
                        s = start + 1
                        e = start + 1 + m
                        if e > vlen:
                            if schemeA_skip_short:
                                continue
                            break
                        gt_win = tgt[b, s:e]
                        pred_win = gen_seq[b, s:e]
                        if schemeA_skip_short:
                            if (gt_win == 0).any() or (pred_win == 0).any():
                                continue
                        windows += 1.0
                        # token accuracy
                        acc = (pred_win == gt_win).float().mean().item()
                        acc_sum += acc
                        # exact
                        exact_sum += 1.0 if (pred_win == gt_win).all().item() else 0.0
                        # overlap
                        overlap_sum += _window_overlap_stats(pred_win, gt_win)
                        # map/ndcg treating GT window as relevant set
                        gt_set = set(gt_win.tolist())
                        ap, nd = _window_map_ndcg(pred_win, gt_set)
                        map_sum += ap
                        ndcg_sum += nd


                if windows > 0:
                    path_aggs[(tag, m)]["acc_sum"] += acc_sum
                    path_aggs[(tag, m)]["exact_sum"] += exact_sum
                    path_aggs[(tag, m)]["overlap_sum"] += overlap_sum
                    path_aggs[(tag, m)]["map_sum"] += map_sum
                    path_aggs[(tag, m)]["ndcg_sum"] += ndcg_sum
                    path_aggs[(tag, m)]["windows"] += windows

    # -------- finalize --------
    results: Dict[str, float] = {}

    for K in k_list:
        denom = step_aggs[("base", K)]["denom"]
        results.update(_finalize_step_metrics(step_aggs[("base", K)], denom, K, prefix="base_"))
        results.update(_finalize_step_metrics(step_aggs[("policy", K)], denom, K, prefix="policy_"))

    for m in m_list:
        for tag in ["base", "policy"]:
            agg = path_aggs[(tag, m)]
            windows = agg["windows"]
            if windows <= 0:
                results[f"{tag}_acc@{m}"] = 0.0
                results[f"{tag}_exact@{m}"] = 0.0
                results[f"{tag}_overlap@{m}"] = 0.0
                results[f"{tag}_map@{m}"] = 0.0
                results[f"{tag}_NDCG@{m}"] = 0.0
                results[f"{tag}_windows@{m}"] = 0.0
                continue
            results[f"{tag}_acc@{m}"] = agg["acc_sum"] / windows
            results[f"{tag}_exact@{m}"] = agg["exact_sum"] / windows
            results[f"{tag}_overlap@{m}"] = agg["overlap_sum"] / windows
            results[f"{tag}_map@{m}"] = agg["map_sum"] / windows
            results[f"{tag}_NDCG@{m}"] = agg["ndcg_sum"] / windows
            results[f"{tag}_windows@{m}"] = windows

    return results

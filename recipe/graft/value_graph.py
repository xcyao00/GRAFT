"""
Value-based Graft: a standalone module implementing step-level credit
assignment via standard RL value estimation on the rollout graph.

Design:

  1. Build a per-task graph from the batch (multigraph: each step = one edge).
  2. Collect ALL final next_obs across the group as terminal sinks, each tagged
     with R(f) in {0, 1} according to whether any reaching trajectory succeeded.
     "Success-wins" merge rule: if a sink is reached by both a success and a
     failure trajectory, treat it as success (R=1).
  3. Estimate V(s) for every node by Bellman value iteration on the empirical
     transition graph. Two aggregation modes are supported:
        - 'action_level' (recommended): V(u) = mean over distinct actions a at u
                                              of  E_v[gamma * V(v)]
        - 'edge_level'                : V(u) = sum_v Phat(v|u) * gamma * V(v)
                                              with Phat estimated from parallel-edge counts
     Sinks are absorbing: V(f) = R(f).
  4. Per executed edge (s -a-> s'), compute step advantage
        r_edge(s, a, s') = gamma * V(s') - V(s)
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

import numpy as np
import torch

from verl import DataProto

# Reuse GiGPO's normalization primitives.
from gigpo.core_gigpo import (
    episode_norm_reward,
    build_step_group,
    step_norm_reward,
)


def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


class StateCanonicalizer:
    """
    Maps similar states to a canonical representative.
    When enabled, states within `threshold` similarity are treated as the same node in the graph.
    """

    def __init__(self, enable: bool = False, threshold: float = 0.95):
        self.enable = enable
        self.threshold = threshold
        # List of (canonical_hashable, canonical_raw_str) pairs
        self._clusters: list = []
        # Cache: raw hashable -> canonical hashable
        self._cache: dict = {}

    def canonicalize(self, state):
        """
        Return the canonical hashable form of `state`.
        If similarity is disabled, falls back to exact `to_hashable`.
        """
        h = to_hashable(state)
        if not self.enable:
            return h

        # Check cache first
        if h in self._cache:
            return self._cache[h]

        raw_str = state if isinstance(state, str) else str(state)

        # Try to find a matching cluster
        for canonical_h, canonical_str in self._clusters:
            if SequenceMatcher(None, raw_str, canonical_str).ratio() >= self.threshold:
                self._cache[h] = canonical_h
                return canonical_h

        # No match — this state becomes a new canonical representative
        self._clusters.append((h, raw_str))
        self._cache[h] = h
        return h


def resolve_advantage_norm_group(
    anchor_obs: np.ndarray,
    index: np.ndarray,
    advantage_norm_scope: str = "node",
    verbose: bool = False,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
):
    """
    Resolve the group-id array used for step-level advantage normalization.

    Args:
        advantage_norm_scope:
            - "node" (default): group = (task uid, anchor_obs). Steps only
              compete against other steps sharing the exact same graph node
              within the same task group (see ``build_step_group``).
            - "task": group = task uid only. Pools ALL steps of ALL
              trajectories in the same task group into one global group,
              avoiding the singleton-zeroing issue at the cost of no longer
              comparing "same-state" actions against each other.

    Returns:
        np.ndarray of shape (bs,): group id for each row.
    """
    if advantage_norm_scope == "node":
        step_group_uids, _ = build_step_group(anchor_obs, index, verbose=verbose,
                                              enable_similarity=enable_similarity,
                                              similarity_thresh=similarity_thresh)
        return step_group_uids
    elif advantage_norm_scope == "task":
        # Pool ALL steps of the same task (uid) into one global group,
        # regardless of anchor_obs. `index` is already the per-row task uid.
        return index
    else:
        raise ValueError(
            f"Unknown advantage_norm_scope: {advantage_norm_scope!r}; expected 'node' or 'task'."
        )


# ---------------------------------------------------------------------------
# Multi-sink graph
# ---------------------------------------------------------------------------
class MultiSinkValueGraph:
    """
    A lightweight per-task graph used only for value estimation.
    """

    def __init__(self):
        # graph[u][v] = list of edge attribute dicts (multigraph)
        self.graph: Dict = defaultdict(lambda: defaultdict(list))
        self.nodes: set = set()
        # sink_hash -> R(f) in {0, 1}
        self.sinks: Dict = {}
        # Optional, for sanity / debugging
        self.first_state = None
        self.canonicalizer: StateCanonicalizer = None

    # ----- construction -----
    def add_edge(self, u, v, **attrs):
        self.graph[u][v].append(attrs)
        self.nodes.add(u)
        self.nodes.add(v)

    @classmethod
    def from_data(
        cls,
        data: DataProto,
        enable_similarity: bool = False,
        similarity_thresh: float = 0.95,
    ) -> Dict[str, "MultiSinkValueGraph"]:
        """
        Build one MultiSinkValueGraph per task uid.

        Sink discovery: for each trajectory in the group, the step with the
        largest data_id (= last step of that rollout) contributes its
        canonicalized next_obs to the sink set, tagged with success label
        derived from this rollout's terminal episode_rewards.

        NOTE: rollout_loop.py rewrites the last next_obs of any successful
        rollout to literal 'success'. So all successful trajectories' sinks
        naturally collapse to canonicalize('success'). Failed trajectories'
        sinks are their real last observations.
        """
        Gs: Dict[str, MultiSinkValueGraph] = {}

        all_uids = set(data.non_tensor_batch['uid'])
        traj_uids_arr = data.non_tensor_batch['traj_uid']
        anchor_obs_arr = data.non_tensor_batch['anchor_obs']
        next_obs_arr = data.non_tensor_batch['next_obs']
        actions_arr = data.non_tensor_batch['text_actions']
        ep_rewards_arr = data.non_tensor_batch['episode_rewards']
        uid_arr = data.non_tensor_batch['uid']

        for uid in all_uids:
            g = cls()
            canon = StateCanonicalizer(enable=enable_similarity, threshold=similarity_thresh)
            uid_indices = np.where(uid_arr == uid)[0]

            # ── add edges ──
            for i in uid_indices:
                anchor_h = canon.canonicalize(anchor_obs_arr[i])
                next_h = canon.canonicalize(next_obs_arr[i])
                g.add_edge(
                    anchor_h,
                    next_h,
                    action=actions_arr[i],
                    traj_uid=traj_uids_arr[i],
                    uid=uid_arr[i],
                    data_id=i,
                    episode_rewards=ep_rewards_arr[i],
                )

            # ── collect sinks: per trajectory, take last step ──
            # ``step_idx`` (stamped by rollout_loop) is the rollout wall-clock
            # time step and is robust to downstream row reordering. If
            # missing, fall back to the global data_id (row index), which
            # matches rollout step order under the current data layout.
            step_idx_arr = data.non_tensor_batch.get('step_idx', None)

            traj_uids_in_uid = set(traj_uids_arr[uid_indices])
            for tuid in traj_uids_in_uid:
                tuid_mask = uid_indices[traj_uids_arr[uid_indices] == tuid]
                if len(tuid_mask) == 0:
                    continue
                if step_idx_arr is not None:
                    last_step = tuid_mask[
                        np.argmax(step_idx_arr[tuid_mask].astype(np.int64))
                    ]
                else:
                    last_step = tuid_mask[np.argmax(tuid_mask)]   # largest data_id
                sink_h = canon.canonicalize(next_obs_arr[last_step])
                # success label for THIS trajectory:
                # episode_rewards is per-step; the last step holds the terminal reward.
                R = 1 if float(ep_rewards_arr[last_step]) > 0 else 0
                # Success-wins merge rule.
                g.sinks[sink_h] = max(g.sinks.get(sink_h, 0), R)
                # Make sure the sink is registered as a node (it always is via add_edge,
                # but be defensive in case of empty edges).
                g.nodes.add(sink_h)

            # ── first_state (best-effort, not used by value iteration) ──
            from collections import Counter
            first_counts = Counter()
            for tuid in traj_uids_in_uid:
                tuid_mask = uid_indices[traj_uids_arr[uid_indices] == tuid]
                if len(tuid_mask) == 0:
                    continue
                if step_idx_arr is not None:
                    first_step = tuid_mask[
                        np.argmin(step_idx_arr[tuid_mask].astype(np.int64))
                    ]
                else:
                    first_step = tuid_mask[np.argmin(tuid_mask)]
                first_counts[canon.canonicalize(anchor_obs_arr[first_step])] += 1
            if first_counts:
                g.first_state = first_counts.most_common(1)[0][0]

            g.canonicalizer = canon
            Gs[uid] = g

        return Gs

    def __repr__(self):
        n_edges = sum(len(self.graph[u][v]) for u in self.graph for v in self.graph[u])
        n_sinks_pos = sum(1 for r in self.sinks.values() if r > 0)
        return (f"MultiSinkValueGraph(nodes={len(self.nodes)}, edges={n_edges}, "
                f"sinks={len(self.sinks)} [success={n_sinks_pos}])")


# ---------------------------------------------------------------------------
# Bellman value iteration
# ---------------------------------------------------------------------------
def compute_bellman_values(
    g: MultiSinkValueGraph,
    gamma: float,
    mode: str = "action_level",
    tol: float = 1e-8,
    max_iters: int = 500,
) -> Dict:
    """
    Bellman value iteration on a MultiSinkValueGraph.

    V(f) = R(f)                                           for sinks f
    V(u) = sum_v Phat(v|u) * gamma * V(v)                 for non-sinks (edge_level)
    V(u) = (1/|A(u)|) * sum_a sum_v Phat(v|u,a) * gamma * V(v)   (action_level)

    Internal-step reward is implicitly 0 (sparse reward MDP, all reward at sinks).

    Implementation notes:
      * Updates are Gauss-Seidel (in-place ``V[u] = new_v``).
      * Scan order is *reverse-BFS distance to any sink* (smallest distance
        first), so information propagates from sinks back to sources in as
        few sweeps as possible. Nodes that cannot reach any sink are scanned
        last (their V stays 0, which is the correct fixed point).
      * Emits a ``warnings.warn`` when ``max_iters`` is exhausted before
        ``delta < tol`` so silent under-convergence is detectable.

    Returns:
        Dict[node_hash, float]: V(s) for every node in the graph.
    """
    if mode not in ("action_level", "edge_level"):
        raise ValueError(f"Unknown mode={mode!r}; expected 'action_level' or 'edge_level'.")

    # Initialize: sinks fixed to R(f); everything else to 0.
    V: Dict = {n: 0.0 for n in g.nodes}
    for sink, R in g.sinks.items():
        if sink in V:
            V[sink] = float(R)

    # Pre-compute transition tables for non-sink nodes.
    # transitions[u] = list of (v, prob) tuples (already mode-aggregated).
    transitions: Dict = {}
    for u in g.graph:
        if u in g.sinks:
            # Sinks are absorbing; no outgoing transitions used in iteration.
            continue
        if mode == "edge_level":
            counts: Dict = defaultdict(int)
            total = 0
            for v in g.graph[u]:
                c = len(g.graph[u][v])
                counts[v] += c
                total += c
            if total == 0:
                continue
            transitions[u] = [(v, counts[v] / total) for v in counts]
        else:  # action_level
            # action -> {next_state: count}
            action_buckets: Dict = defaultdict(lambda: defaultdict(int))
            for v in g.graph[u]:
                for edge_attr in g.graph[u][v]:
                    a = edge_attr.get('action', None)
                    action_buckets[a][v] += 1
            num_actions = len(action_buckets)
            if num_actions == 0:
                continue
            agg: Dict = defaultdict(float)
            for a, next_dist in action_buckets.items():
                total_a = sum(next_dist.values())
                if total_a == 0:
                    continue
                for v, c in next_dist.items():
                    agg[v] += (c / total_a) / num_actions
            transitions[u] = list(agg.items())

    # ── Build a reverse-BFS order: distance from each node to ANY sink ──
    # Predecessors map: preds[v] = set of u s.t. u -> v exists.
    preds: Dict = defaultdict(set)
    for u in g.graph:
        for v in g.graph[u]:
            preds[v].add(u)

    INF = 10 ** 9
    dist: Dict = {s: 0 for s in g.sinks}
    queue = deque(g.sinks.keys())
    while queue:
        v = queue.popleft()
        for u in preds.get(v, ()):  # walk edges backward
            if u not in dist:
                dist[u] = dist[v] + 1
                queue.append(u)
    # Stable order: nodes with smaller distance to a sink are scanned first;
    # unreachable nodes get INF and end up at the tail.
    scan_order: List = sorted(
        transitions.keys(), key=lambda u: dist.get(u, INF)
    )

    # Value iteration. Because gamma < 1 this is a contraction; converges fast.
    delta = 0.0
    converged = False
    for _ in range(max_iters):
        delta = 0.0
        for u in scan_order:
            succ = transitions[u]
            new_v = 0.0
            for v, p in succ:
                new_v += p * gamma * V.get(v, 0.0)
            diff = abs(new_v - V[u])
            if diff > delta:
                delta = diff
            V[u] = new_v
        if delta < tol:
            converged = True
            break

    if not converged and len(scan_order) > 0:
        warnings.warn(
            f"compute_bellman_values did not converge: final delta={delta:.3e} "
            f"> tol={tol:.0e} after {max_iters} iters "
            f"(gamma={gamma}, |V|={len(V)}, |non_sink|={len(scan_order)}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return V


def _adaptive_max_iters(gamma: float, tol: float, user_max_iters: int) -> int:
    """
    Pick a max_iters that is large enough for value iteration to actually
    reach ``tol`` given the discount factor ``gamma``.

    Value iteration error decays as ``gamma^k``; we need
        gamma^k * V_range  <=  tol
    With ``V_range <= 1`` (sinks are in {0, 1}) the worst-case bound is
        k_required = ceil(log(tol) / log(gamma))
    Add a small safety margin (+50). When ``gamma`` is tiny (<= 0.1) the
    user-supplied ``user_max_iters`` already vastly exceeds what's needed,
    so we never *shrink* it below the user value.
    """
    if not (0.0 < gamma < 1.0):
        # gamma == 0 → one sweep suffices; gamma >= 1 is not a contraction.
        return user_max_iters
    if tol <= 0.0:
        return user_max_iters
    try:
        k_required = int(math.ceil(math.log(tol) / math.log(gamma))) + 50
    except (ValueError, ZeroDivisionError):
        return user_max_iters
    # Never go below the user-supplied floor; cap to a sane upper bound to
    # avoid pathological cases (gamma extremely close to 1).
    return max(user_max_iters, min(k_required, 200_000))


def compute_all_bellman_values(
    Gs: Dict[str, MultiSinkValueGraph],
    gamma: float,
    mode: str = "action_level",
    tol: float = 1e-8,
    max_iters: int = 500,
) -> Dict[str, Dict]:
    """
    Convenience wrapper: run compute_bellman_values for every task uid.

    Adaptive ``max_iters``: the user-supplied value is treated as a *floor*;
    if ``gamma`` is large enough that value iteration provably needs more
    sweeps to reach ``tol``, we automatically raise it (see
    ``_adaptive_max_iters``). This prevents silent under-convergence when
    the caller bumps ``gamma`` (e.g. 0.1 → 0.99) without retuning
    ``max_iters``.

    Returns: Dict[uid, Dict[node_hash, float]]
    """
    effective_max_iters = _adaptive_max_iters(gamma, tol, max_iters)
    return {
        uid: compute_bellman_values(
            g, gamma=gamma, mode=mode, tol=tol, max_iters=effective_max_iters
        )
        for uid, g in Gs.items()
    }


# ---------------------------------------------------------------------------
# Step advantage = gamma * V(s') - V(s)
# ---------------------------------------------------------------------------
def compute_value_based_step_advantages(
    batch: DataProto,
    Gs: Dict[str, MultiSinkValueGraph],
    values_per_uid: Dict[str, Dict],
    gamma: float,
) -> torch.Tensor:
    """
    Compute per-step "advantage" = gamma * V(next_obs) - V(anchor_obs).

    Returns:
        torch.Tensor of shape (bs,) on the same device as batch.batch['input_ids'].
    """
    traj_uids = batch.non_tensor_batch['traj_uid']
    uids = batch.non_tensor_batch['uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    next_obs_ = batch.non_tensor_batch['next_obs']
    old_obs_ = batch.non_tensor_batch['anchor_obs']

    bs = len(traj_uids)
    out = np.zeros(bs, dtype=np.float32)

    unique_traj_uids = np.unique(traj_uids)
    for tuid in unique_traj_uids:
        traj_indices = np.where(traj_uids == tuid)[0]
        uid_uid = uids[traj_indices[0]]

        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s within a trajectory"

        g = Gs[uid_uid]
        V = values_per_uid[uid_uid]
        canon = getattr(g, 'canonicalizer', None)

        for i in traj_indices:
            if canon is not None:
                old_h = canon.canonicalize(old_obs_[i])
                new_h = canon.canonicalize(next_obs_[i])
            else:
                old_h = to_hashable(old_obs_[i])
                new_h = to_hashable(next_obs_[i])
            v_old = V.get(old_h, 0.0)
            v_new = V.get(new_h, 0.0)
            out[i] = gamma * v_new - v_old

    return torch.tensor(out, dtype=torch.float32, device=batch.batch['input_ids'].device)


# ---------------------------------------------------------------------------
# Top-level entry: build graphs + values + step advantages in one shot.
# ---------------------------------------------------------------------------
def build_value_graph_and_step_advantages(
    batch: DataProto,
    gamma: float,
    mode: str = "action_level",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    tol: float = 1e-8,
    max_iters: int = 500,
) -> Tuple[Dict[str, MultiSinkValueGraph], Dict[str, Dict], torch.Tensor]:
    """
    One-shot pipeline: from a batch, produce (Gs, values_per_uid, step_advantages).

    Use this if you do not need to keep the graphs / values for inspection;
    otherwise call the three building blocks individually.
    """
    Gs = MultiSinkValueGraph.from_data(
        batch,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    values_per_uid = compute_all_bellman_values(
        Gs, gamma=gamma, mode=mode, tol=tol, max_iters=max_iters
    )
    step_advantages = compute_value_based_step_advantages(
        batch, Gs, values_per_uid, gamma=gamma
    )
    return Gs, values_per_uid, step_advantages


def compute_value_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    episode_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    advantage_norm_scope: str = "node",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
):
    if mode == "none":
        # No normalization: use raw Bellman step rewards directly as advantage.
        step_advantages = step_rewards.unsqueeze(-1).expand_as(response_mask) * response_mask
        if episode_advantage_w != 0.0:
            episode_advantages = episode_norm_reward(
                token_level_rewards, response_mask, index, traj_index, epsilon, remove_std=False
            )
            scores = step_advantage_w * step_advantages + episode_advantage_w * episode_advantages
        else:
            scores = step_advantage_w * step_advantages
        return scores, scores

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    step_group_uids = resolve_advantage_norm_group(anchor_obs, index, advantage_norm_scope, verbose=False,
                                                   enable_similarity=enable_similarity,
                                                   similarity_thresh=similarity_thresh)
    step_advantages = step_norm_reward(
        step_rewards, response_mask, step_group_uids, epsilon, remove_std
    )

    if episode_advantage_w != 0.0:
        episode_advantages = episode_norm_reward(
            token_level_rewards, response_mask, index, traj_index, epsilon, remove_std
        )
        scores = step_advantage_w * step_advantages + episode_advantage_w * episode_advantages
    else:
        scores = step_advantage_w * step_advantages

    return scores, scores


# ---------------------------------------------------------------------------
# multi-hop edge-global GAE
# ---------------------------------------------------------------------------
def compute_multihop_edge_gae(
    g: MultiSinkValueGraph,
    V: Dict,
    gamma: float,
    lam: float,
    start_node,
    tol: float = 1e-5,
    max_hops: int = 20,
    raw_edges: Dict = None,
) -> float:
    """
    Compute the multi-hop edge-global GAE value G(start_node).

    G(s1) = sum_{k=1}^{inf} (gamma*lam)^k * TD(k)

    where TD(k) = mean over all raw edge instances reachable from s1 in
    exactly k hops of [gamma*V(v) - V(u)].

    Note: TD(0) (the specific executed edge s0->s1) is NOT included here;
    the caller adds it separately:
        A(s0, a, s1) = [gamma*V(s1) - V(s0)] + G(s1)

    Returns 0.0 if start_node has no outgoing edges (sink or dead-end).

    Cycle handling
    --------------
    In graphs with cycles (e.g. a "go back" action revisiting an earlier
    state), the path-instance frontier can grow exponentially. Two
    complementary safeguards are applied:

    1. ``max_hops`` hard cap: the BFS is truncated after at most ``max_hops``
       iterations regardless of whether the frontier is empty or the weight
       has decayed below ``tol``.

    2. Cycle detection via visited-node deduplication: after each hop we
       record the SET of distinct nodes in the frontier. If the frontier's
       node set is a subset of all previously visited nodes (i.e. no new
       node is being reached), the BFS has entered a pure cycle and will
       never discover new states -- we break early.
    """
    if lam == 0.0:
        return 0.0, 0

    # Pre-build raw edge list for every non-sink node.
    # We need the full graph's raw edges, not just start_node's.
    # Build lazily: raw_edges[u] = list of v for each raw edge instance u->v.
    # If raw_edges is provided externally (pre-built per uid), reuse it directly.
    if raw_edges is None:
        raw_edges = {}
        for u in g.graph:
            if u in g.sinks:
                continue
            flat: List = []
            for v, edge_list in g.graph[u].items():
                flat.extend([v] * len(edge_list))
            if flat:
                raw_edges[u] = flat

    if start_node not in raw_edges:
        return 0.0, 0

    # BFS layer by layer from start_node.
    # frontier: set of unique nodes reachable at current hop.
    # TD(k) averages over all edge instances leaving frontier nodes,
    # so raw_edges[u] (with duplicates) is iterated directly for td_sum.
    # next_frontier is a set to keep size bounded by graph node count.
    frontier: set = {start_node}
    weight = gamma * lam   # (gamma*lam)^k for k=1 at first iteration
    result = 0.0
    visited_nodes: set = {start_node}   # all distinct nodes ever in frontier
    hop = 0
    max_frontier_size = 0

    while frontier and weight > tol and hop < max_hops:
        next_frontier: set = set()
        td_sum = 0.0
        edge_count = 0
        for u in frontier:
            edges_u = raw_edges.get(u, [])
            v_u = V.get(u, 0.0)
            for v in edges_u:
                if v == u:  # skip self-loops
                    continue
                td_sum += gamma * V.get(v, 0.0) - v_u
                edge_count += 1
                if v in raw_edges:  # only non-sink nodes enter next frontier
                    next_frontier.add(v)

        if edge_count == 0:
            break

        result += weight * (td_sum / edge_count)

        frontier = next_frontier
        if len(frontier) > max_frontier_size:
            max_frontier_size = len(frontier)
        weight *= gamma * lam
        hop += 1

        # Cycle detection: if no new node is discovered, break early.
        new_nodes = frontier - visited_nodes
        if not new_nodes:
            break
        visited_nodes.update(new_nodes)

    return result, max_frontier_size


def compute_multihop_edge_gae_action_level(
    g: MultiSinkValueGraph,
    V: Dict,
    gamma: float,
    lam: float,
    start_node,
    tol: float = 1e-5,
    max_hops: int = 20,
    raw_edges_action: Dict = None,
) -> float:
    """
    Action-level variant of compute_multihop_edge_gae (Scheme D).

    G(s1) = sum_{k=1}^{inf} (gamma*lam)^k * TD_action(k)

    TD_action(k) is computed as follows for the k-hop frontier reachable
    from start_node:

      1. Collect all raw edge instances (u, action, v) reachable in exactly
         k hops (edges can span multiple distinct (u,v) node pairs).
      2. Group by (u, action): within each group, average over the distinct
         next-states v weighted by their empirical transition probability
         hat_P(v|u,a) = count(u->a->v) / sum_v' count(u->a->v').
         This gives TD_bar(u,a) = sum_v hat_P(v|u,a)*[gamma*V(v)-V(u)].
      3. Average uniformly over all unique (u, action) pairs in the frontier:
         TD_action(k) = mean_{(u,a) in UA_k} TD_bar(u,a).

    This mirrors the action-level Bellman weighting but applied globally
    across all nodes reachable at hop k (not restricted to a single node
    pair as in the Bellman iteration).

    Note: TD(0) is NOT included here; the caller adds it separately.

    Returns 0.0 if start_node has no outgoing edges (sink or dead-end).
    """
    if lam == 0.0:
        return 0.0

    # Build action-aware edge map: raw_edges_action[u] = list of (action, v)
    # for each raw edge instance leaving u.
    # If raw_edges_action is provided externally (pre-built per uid), reuse it directly.
    if raw_edges_action is None:
        raw_edges_action = {}
        for u in g.graph:
            if u in g.sinks:
                continue
            flat: List = []
            for v, edge_list in g.graph[u].items():
                for attr in edge_list:
                    flat.append((attr.get('action', None), v))
            if flat:
                raw_edges_action[u] = flat

    if start_node not in raw_edges_action:
        return 0.0, 0

    # BFS layer by layer from start_node.
    # frontier: list of nodes at the current hop (one entry per path instance).
    frontier: List = [start_node]
    weight = gamma * lam
    result = 0.0
    visited_nodes: set = {start_node}
    hop = 0

    while frontier and weight > tol and hop < max_hops:
        # Step 1: collect all (u, action, v) triples from the frontier.
        # Use a dict: (u, action) -> {v: count} to aggregate.
        ua_to_v_counts: Dict = {}
        next_frontier: List = []

        for u in frontier:
            edges_u = raw_edges_action.get(u, [])
            for action, v in edges_u:
                if v == u:  # skip self-loops to avoid exponential frontier blowup
                    continue
                key = (u, action)
                if key not in ua_to_v_counts:
                    ua_to_v_counts[key] = {}
                ua_to_v_counts[key][v] = ua_to_v_counts[key].get(v, 0) + 1
                next_frontier.append(v)

        if not ua_to_v_counts:
            break

        # Step 2: compute TD_bar(u, a) for each unique (u, action) pair.
        td_sum = 0.0
        ua_count = 0
        for (u, action), v_counts in ua_to_v_counts.items():
            total = sum(v_counts.values())
            v_u = V.get(u, 0.0)
            # hat_P(v|u,a) * [gamma*V(v) - V(u)]
            td_bar = sum(
                (cnt / total) * (gamma * V.get(v, 0.0) - v_u)
                for v, cnt in v_counts.items()
            )
            td_sum += td_bar
            ua_count += 1

        # Step 3: uniform mean over all unique (u, action) pairs.
        td_mean = td_sum / ua_count
        result += weight * td_mean

        frontier = [v for v in next_frontier if v in raw_edges_action]
        weight *= gamma * lam
        hop += 1

        new_nodes = set(frontier) - visited_nodes
        if not new_nodes:
            break
        visited_nodes.update(new_nodes)

    return result, 0  # action_level does not track max_frontier_size


def compute_gae_graph_multihop_step_advantages(
    batch: DataProto,
    Gs: Dict[str, MultiSinkValueGraph],
    values_per_uid: Dict[str, Dict],
    gamma: float,
    lam: float,
    tol: float = 1e-5,
    td_weighting: str = "edge",
) -> torch.Tensor:
    """
    For each executed edge (s0, a, s1) in the batch, compute:
        A(s0, a, s1) = TD(0) + G(s1)
        TD(0)        = gamma * V(s1) - V(s0)
        G(s1)        = sum_{k=1}^{inf} (gamma*lam)^k * TD(k)

    ``td_weighting`` controls how TD(k) is computed for k >= 1:
      - ``"edge"`` (default): uniform mean over all raw edge instances
        reachable from s1 in exactly k hops (every edge instance gets
        equal weight 1/|E_k|). Equivalent to edge-level weighting.
      - ``"action"``: action-level weighting. Edges are first grouped by
        (u, action); within each group hat_P(v|u,a) is estimated from
        empirical counts; then TD_bar(u,a) is computed per group; finally
        a uniform mean is taken over all unique (u, action) pairs in the
        k-hop frontier. This mirrors the action-level Bellman weighting
        applied globally across all reachable nodes at each hop.

    G(s1) is cached per (uid, s1) to avoid redundant BFS traversals when
    multiple executed edges share the same s1.

    Returns
    -------
    torch.Tensor of shape (bs,) on the same device as batch.batch['input_ids'].
    """
    if td_weighting not in ("edge", "action"):
        raise ValueError(f"td_weighting must be 'edge' or 'action', got {td_weighting!r}")

    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    traj_uids = batch.non_tensor_batch['traj_uid']
    uids = batch.non_tensor_batch['uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    next_obs_ = batch.non_tensor_batch['next_obs']
    old_obs_ = batch.non_tensor_batch['anchor_obs']

    bs = len(traj_uids)
    out = np.zeros(bs, dtype=np.float32)

    # Collect all (tuid, traj_indices, uid_uid) tuples, one per trajectory.
    all_trajs: List = []
    for tuid in np.unique(traj_uids):
        traj_indices = np.where(traj_uids == tuid)[0]
        uid_uid = uids[traj_indices[0]]
        all_trajs.append((tuid, traj_indices, uid_uid))

    def _process_traj(tuid, traj_indices, uid_uid) -> List:
        """Compute GAE advantages for one trajectory.
        """
        g = Gs[uid_uid]
        V = values_per_uid[uid_uid]
        canon = getattr(g, 'canonicalizer', None)

        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s within a trajectory"

        # Build raw_edges independently per trajectory (read-only access to g).
        if td_weighting == "action":
            raw_edges: Dict = {}
            for u in g.graph:
                if u in g.sinks:
                    continue
                flat: List = []
                for v, edge_list in g.graph[u].items():
                    for attr in edge_list:
                        flat.append((attr.get('action', None), v))
                if flat:
                    raw_edges[u] = flat
        else:
            raw_edges = {}
            for u in g.graph:
                if u in g.sinks:
                    continue
                flat = []
                for v, edge_list in g.graph[u].items():
                    flat.extend([v] * len(edge_list))
                if flat:
                    raw_edges[u] = flat

        _n_nodes = len(g.graph)
        _n_edges = sum(
            sum(len(el) for el in nbrs.values())
            for nbrs in g.graph.values()
        )

        # G_cache is local to this trajectory, no sharing across threads.
        local_G_cache: Dict = {}
        results: List = []

        _traj_t0 = _time.perf_counter()
        for i in traj_indices:
            if canon is not None:
                old_h = canon.canonicalize(old_obs_[i])
                new_h = canon.canonicalize(next_obs_[i])
            else:
                old_h = to_hashable(old_obs_[i])
                new_h = to_hashable(next_obs_[i])

            v_old = V.get(old_h, 0.0)
            v_new = V.get(new_h, 0.0)
            td0 = gamma * v_new - v_old

            if new_h not in local_G_cache:
                _bfs_t0 = _time.perf_counter()
                if td_weighting == "action":
                    local_G_cache[new_h], _max_fs = compute_multihop_edge_gae_action_level(
                        g, V, gamma=gamma, lam=lam, start_node=new_h,
                        tol=tol, raw_edges_action=raw_edges,
                    )
                else:
                    local_G_cache[new_h], _max_fs = compute_multihop_edge_gae(
                        g, V, gamma=gamma, lam=lam, start_node=new_h,
                        tol=tol, raw_edges=raw_edges,
                    )
                _bfs_t1 = _time.perf_counter()
                # print(
                #     f"[multihop_gae_bfs] node={str(new_h)[:30]!r} "
                #     f"max_frontier={_max_fs} time={_bfs_t1 - _bfs_t0:.4f}s",
                #     flush=True,
                # )
            results.append((i, td0 + local_G_cache[new_h]))

        _traj_t1 = _time.perf_counter()
        # print(
        #     f"[multihop_gae] traj={tuid} traj_len={len(traj_indices)} "
        #     f"graph_nodes={_n_nodes} graph_edges={_n_edges} "
        #     f"time={_traj_t1 - _traj_t0:.4f}s",
        #     flush=True,
        # )
        return results

    with ThreadPoolExecutor(max_workers=min(len(all_trajs), 64)) as executor:
        futures = [executor.submit(_process_traj, tuid, traj_indices, uid_uid)
                   for tuid, traj_indices, uid_uid in all_trajs]
        for future in futures:
            for idx, val in future.result():
                out[idx] = val

    return torch.tensor(out, dtype=torch.float32, device=batch.batch['input_ids'].device)


def build_value_graph_and_gae_step_advantages(
    batch: DataProto,
    gamma: float,
    lam: float,
    bellman_mode: str = "action_level",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    tol: float = 1e-8,
    max_iters: int = 500,
) -> Tuple[Dict[str, MultiSinkValueGraph], Dict[str, Dict], torch.Tensor]:
    """
    One-shot pipeline for GAE-based step advantages.

    Returns
    -------
    (Gs, values_per_uid, step_adv)
        Gs               : per-uid MultiSinkValueGraph
        values_per_uid   : per-uid V(s) map (Bellman fixed point)
        step_adv         : torch.Tensor of shape (bs,); to be broadcast over
                           response_mask by the trainer.
    """
    Gs = MultiSinkValueGraph.from_data(
        batch,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    values_per_uid = compute_all_bellman_values(
        Gs, gamma=gamma, mode=bellman_mode, tol=tol, max_iters=max_iters,
    )
   
    step_adv = compute_gae_graph_multihop_step_advantages(
        batch, Gs, values_per_uid, gamma=gamma, lam=lam, tol=tol,
    )

    # ── Information Propagation Degree statistics ──
    _all_prop_degrees: List = []
    for uid, g in Gs.items():
        # direct_trajs[u]: traj_uids directly on u's outgoing edges
        _direct: Dict = {}
        for u in g.graph:
            _tset: set = set()
            for v, edge_list in g.graph[u].items():
                for edge_attr in edge_list:
                    _tset.add(edge_attr['traj_uid'])
            if _tset:
                _direct[u] = _tset

        # Step 2: propagate reachable traj_uid sets in reverse-BFS order.
        # Use the same scan_order as Bellman (small dist-to-sink first),
        # but iterate in REVERSE (source-first) for forward propagation.
        # Build predecessor map for BFS distance (reuse Bellman's preds logic).
        _preds: Dict = defaultdict(set)
        for u in g.graph:
            for v in g.graph[u]:
                _preds[v].add(u)
        _INF = 10 ** 9
        _dist: Dict = {s: 0 for s in g.sinks}
        _q: deque = deque(g.sinks.keys())
        while _q:
            _v = _q.popleft()
            for _u in _preds.get(_v, ()):
                if _u not in _dist:
                    _dist[_u] = _dist[_v] + 1
                    _q.append(_u)
        # scan_order: nodes sorted by DESCENDING dist (sources first)
        _scan_order = sorted(
            [u for u in g.nodes if u not in g.sinks],
            key=lambda u: -_dist.get(u, _INF),
        )

        # reachable[u] = set of traj_uids reachable from u (including via successors)
        _reachable: Dict = {}
        # Iterate until convergence (handles cycles)
        for _iter in range(200):
            _changed = False
            for u in _scan_order:
                _new_set = set(_direct.get(u, set()))
                for v in g.graph.get(u, {}):
                    _new_set |= _reachable.get(v, set())
                _old = _reachable.get(u)
                if _old != _new_set:
                    _reachable[u] = _new_set
                    _changed = True
            if not _changed:
                break

        for u in _scan_order:
            _deg = len(_reachable.get(u, set()))
            if _deg > 0:
                _all_prop_degrees.append(_deg)

    if _all_prop_degrees:
        _pd_avg = float(np.mean(_all_prop_degrees))
        _pd_min = int(min(_all_prop_degrees))
        _pd_max = int(max(_all_prop_degrees))
        _pd_total = len(_all_prop_degrees)
        _pd_bucket_defs = [
            ('1', 1, 1), ('2', 2, 2), ('3', 3, 3), ('4', 4, 4),
            ('5', 5, 5), ('6', 6, 6), ('7', 7, 7), ('8', 8, 8),
        ]
        _pd_parts = []
        for _label, _lo, _hi in _pd_bucket_defs:
            _cnt = sum(1 for d in _all_prop_degrees if _lo <= d <= _hi)
            _pd_parts.append(f"{_label}:{100.0*_cnt/_pd_total:.1f}%")
        print(
            f"[PropDegree] avg={_pd_avg:.4f} min={_pd_min} max={_pd_max} "
            f"num_nodes={_pd_total} | " + " ".join(_pd_parts)
        )

    return Gs, values_per_uid, step_adv


def compute_gae_outcome_advantage(
    step_advantages_scalar: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    episode_advantage_w: float = 0.0,
    token_level_rewards: torch.Tensor = None,
    traj_index: np.ndarray = None,
    mode: str = "mean_std_norm",
    advantage_norm_scope: str = "node",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
):
    """
    Turn per-step scalar GAE advantages into (bs, response_length) advantages
    to feed the actor loss.

    Parameters
    ----------
    step_advantages_scalar : torch.Tensor, shape (bs,)
        Per-step GAE advantage (either scheme A or scheme B).
    response_mask : torch.Tensor, shape (bs, response_length)
    anchor_obs, index : np.ndarray of shape (bs,)
        Used to build the anchor-state step group when ``mode`` normalises.
    episode_advantage_w : float
        Weight for the episode-level advantage term. Set to 0.0 (default) to
        disable it entirely (original behaviour). Requires
        ``token_level_rewards`` and ``traj_index`` when non-zero.
    token_level_rewards : torch.Tensor, shape (bs, response_length), optional
        Token-level rewards used to compute the episode-level advantage.
        Required when ``episode_advantage_w != 0.0``.
    traj_index : np.ndarray of shape (bs,), optional
        Trajectory uid for each sample, used to group episodes.
        Required when ``episode_advantage_w != 0.0``.
    mode : {"none", "mean_norm", "mean_std_norm"}
        - "none"         : broadcast raw GAE values over response tokens.
        - "mean_norm"    : subtract per-anchor-group mean.
        - "mean_std_norm": subtract mean and divide by std within each group.
    advantage_norm_scope : {"node", "task"}
    step_advantage_w : float
        Final scale factor applied to the (normalised) step advantage.

    Returns
    -------
    (scores, scores) : (torch.Tensor, torch.Tensor)
        Both shape (bs, response_length). Second slot mirrors ``returns`` in
        the existing pipeline; PPO/GRPO trainers here treat it identically to
        ``advantages``.
    """
    if mode == "none":
        step_adv_expanded = step_advantages_scalar.unsqueeze(-1).expand_as(response_mask) * response_mask
        if episode_advantage_w != 0.0:
            assert token_level_rewards is not None and traj_index is not None, (
                "token_level_rewards and traj_index must be provided when episode_advantage_w != 0.0"
            )
            episode_advantages = episode_norm_reward(
                token_level_rewards, response_mask, index, traj_index, epsilon, remove_std=False
            )
            scores = step_advantage_w * step_adv_expanded + episode_advantage_w * episode_advantages
        else:
            scores = step_advantage_w * step_adv_expanded
        return scores, scores

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    group_uids = resolve_advantage_norm_group(anchor_obs, index, advantage_norm_scope, verbose=False,
                                              enable_similarity=enable_similarity,
                                              similarity_thresh=similarity_thresh)

    step_adv_normed = step_norm_reward(
        step_advantages_scalar, response_mask, group_uids, epsilon, remove_std,
    )

    if episode_advantage_w != 0.0:
        assert token_level_rewards is not None and traj_index is not None, (
            "token_level_rewards and traj_index must be provided when episode_advantage_w != 0.0"
        )
        episode_advantages = episode_norm_reward(
            token_level_rewards, response_mask, index, traj_index, epsilon, remove_std
        )
        scores = step_advantage_w * step_adv_normed + episode_advantage_w * episode_advantages
    else:
        scores = step_advantage_w * step_adv_normed

    return scores, scores


# ── graph data persistence ────────────────────────────────────────────────────

def save_graph_data(
    Gs: Dict[str, "MultiSinkValueGraph"],
    values_per_uid: Dict[str, Dict],
    save_path: str,
):
    """
    Serialize graph data (nodes, edges, V values) to a JSON file.

    Output format per uid:
    {
      "uid": {
        "nodes": [
          {"id": <hash_str>, "state": <obs_str>, "V": <float>, "is_sink": <bool>, "sink_R": <float|null>}
        ],
        "edges": [
          {"src": <hash_str>, "dst": <hash_str>, "action": <str>, "traj_uid": <str>, "count": <int>}
        ]
      }
    }
    Edges with the same (src, dst, action) are aggregated; count = number of instances.
    """
    import json
    import os

    data = {}
    for uid, g in Gs.items():
        V = values_per_uid.get(uid, {})

        # ── nodes ──
        nodes = []
        for node in g.nodes:
            nodes.append({
                "id":      str(node),
                "state":   str(node),
                "V":       float(V.get(node, 0.0)),
                "is_sink": node in g.sinks,
                "sink_R":  float(g.sinks[node]) if node in g.sinks else None,
            })

        # ── edges: aggregate by (src, dst, action) ──
        edge_agg: Dict = {}   # (src_str, dst_str, action_str) -> count
        for src in g.graph:
            for dst, edge_list in g.graph[src].items():
                for e in edge_list:
                    action = str(e.get("action", ""))
                    traj   = str(e.get("traj_uid", ""))
                    key    = (str(src), str(dst), action)
                    if key not in edge_agg:
                        edge_agg[key] = {"count": 0, "traj_uids": set()}
                    edge_agg[key]["count"] += 1
                    edge_agg[key]["traj_uids"].add(traj)

        edges = []
        for (src, dst, action), info in edge_agg.items():
            edges.append({
                "src":      src,
                "dst":      dst,
                "action":   action,
                "count":    info["count"],
                "traj_uids": sorted(info["traj_uids"]),
            })

        data[uid] = {"nodes": nodes, "edges": edges}

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return save_path

"""
Basic Influence Maximization algorithms on static graphs.

Implements the core pieces from Kempe, Kleinberg & Tardos (KDD 2003):
  - Independent Cascade (IC) diffusion simulation
  - Monte Carlo spread estimation  sigma(A)
  - Naive Greedy hill-climbing (with the (1 - 1/e) guarantee)
  - CELF (Cost-Effective Lazy Forward) -- same output as Greedy, much faster,
    using submodularity to avoid recomputing marginal gains for every node
    at every step (Leskovec et al., 2007)
  - Baseline heuristics: High-Degree, Random

Diffusion model used: "Weighted Cascade" (a parameter-free variant of IC
also used in the original paper) -- edge (u, v) succeeds with probability
1 / in-degree(v). This avoids having to hand-pick a propagation probability.
"""

import heapq
import random
import time
import networkx as nx


# ---------------------------------------------------------------------------
# Diffusion model
# ---------------------------------------------------------------------------

def build_propagation_probs(G):
    """Weighted Cascade: edge (u, v) activates v with probability 1/in-degree(v)."""
    probs = {}
    for u, v in G.edges():
        indeg = G.in_degree(v) if G.is_directed() else G.degree(v)
        probs[(u, v)] = 1.0 / indeg if indeg > 0 else 0.0
        if not G.is_directed():
            probs[(v, u)] = 1.0 / (G.in_degree(u) if G.is_directed() else G.degree(u))
    return probs


def simulate_ic(G, seeds, probs, rng):
    """One Monte Carlo run of Independent Cascade starting from `seeds`.
    Returns the number of nodes activated by the end of the cascade."""
    active = set(seeds)
    frontier = list(seeds)
    while frontier:
        new_frontier = []
        for u in frontier:
            neighbors = G.successors(u) if G.is_directed() else G.neighbors(u)
            for v in neighbors:
                if v in active:
                    continue
                p = probs.get((u, v), 0.0)
                if rng.random() < p:
                    active.add(v)
                    new_frontier.append(v)
        frontier = new_frontier
    return len(active)


def estimate_spread(G, seeds, probs, R, rng):
    """Monte Carlo estimate of sigma(seeds): average activated count over R runs."""
    if not seeds:
        return 0.0
    total = 0
    for _ in range(R):
        total += simulate_ic(G, seeds, probs, rng)
    return total / R


# ---------------------------------------------------------------------------
# Naive Greedy (Kempe, Kleinberg & Tardos 2003)
# ---------------------------------------------------------------------------

def greedy(G, k, probs, R=200, seed=0):
    """Naive greedy hill-climbing. At each step, add the node with the
    largest marginal gain in expected spread, estimated by Monte Carlo.
    Guarantees (1 - 1/e) approximation since sigma(.) is submodular."""
    rng = random.Random(seed)
    S = []
    spread_history = [0.0]
    n_spread_evals = 0
    for _ in range(k):
        best_node, best_gain = None, -1.0
        base = estimate_spread(G, S, probs, R, rng)
        for v in G.nodes():
            if v in S:
                continue
            n_spread_evals += 1
            gain = estimate_spread(G, S + [v], probs, R, rng) - base
            if gain > best_gain:
                best_gain, best_node = gain, v
        S.append(best_node)
        spread_history.append(base + best_gain)
    return S, spread_history, n_spread_evals


# ---------------------------------------------------------------------------
# CELF (Leskovec et al. 2007) -- identical output to greedy, far fewer evals
# ---------------------------------------------------------------------------

def celf(G, k, probs, R=200, seed=0):
    """CELF: exploits submodularity. A node's marginal gain can only shrink
    as the seed set grows, so we lazily re-evaluate only the top candidate
    instead of every node at every iteration."""
    rng = random.Random(seed)
    n_spread_evals = 0

    # initial round: evaluate marginal gain of every single node
    gains = []
    for v in G.nodes():
        n_spread_evals += 1
        g = estimate_spread(G, [v], probs, R, rng)
        heapq.heappush(gains, (-g, v, 0))  # (neg gain, node, "last recomputed at" iteration)

    S = []
    spread_history = [0.0]
    current_spread = 0.0

    while len(S) < k:
        neg_gain, v, last_computed = heapq.heappop(gains)
        if last_computed == len(S):
            # gain is up to date -> safe to add
            S.append(v)
            current_spread += -neg_gain
            spread_history.append(current_spread)
        else:
            # recompute marginal gain fresh, push back, and retry
            n_spread_evals += 1
            new_spread = estimate_spread(G, S + [v], probs, R, rng)
            new_gain = new_spread - current_spread
            heapq.heappush(gains, (-new_gain, v, len(S)))

    return S, spread_history, n_spread_evals


# ---------------------------------------------------------------------------
# Baseline heuristics
# ---------------------------------------------------------------------------

def high_degree(G, k):
    deg = dict(G.out_degree() if G.is_directed() else G.degree())
    return sorted(deg, key=deg.get, reverse=True)[:k]


def random_nodes(G, k, seed=0):
    rng = random.Random(seed)
    return rng.sample(list(G.nodes()), k)


# ---------------------------------------------------------------------------
# Evaluation helper: spread curve for a fixed seed ORDER (e.g. from degree/random)
# ---------------------------------------------------------------------------

def spread_curve_for_ordering(G, ordering, probs, R, seed=0):
    rng = random.Random(seed)
    history = [0.0]
    for i in range(1, len(ordering) + 1):
        history.append(estimate_spread(G, ordering[:i], probs, R, rng))
    return history
"""
Synthetic dynamic social network generator.

Since we don't have labeled real-world dynamic-cascade data on hand, we
simulate a plausible dynamic network: start from a scale-free (Barabasi-
Albert) graph, then evolve it over T snapshots by (a) adding new edges via
preferential attachment (new ties form more often to already well-connected
nodes -- mimics real social network growth) and (b) randomly decaying a
small fraction of existing edges (relationships fading). Node set is fixed
across time; only the edge set changes.
"""

import random
import networkx as nx


def make_initial_graph(n=100, m=3, seed=0):
    return nx.barabasi_albert_graph(n, m, seed=seed)


def evolve_graph(G, add_frac=0.06, remove_frac=0.04, rng=None):
    """Return a NEW graph: G with some edges removed (decay) and some
    new edges added (preferential attachment growth)."""
    rng = rng or random.Random()
    G2 = G.copy()
    n_edges = G2.number_of_edges()

    # --- decay: remove a random subset of existing edges ---
    n_remove = max(1, int(remove_frac * n_edges))
    edges = list(G2.edges())
    for u, v in rng.sample(edges, min(n_remove, len(edges))):
        G2.remove_edge(u, v)

    # --- growth: add new edges via (soft) preferential attachment ---
    n_add = max(1, int(add_frac * n_edges))
    nodes = list(G2.nodes())
    degrees = dict(G2.degree())
    weights = [degrees[v] + 1 for v in nodes]  # +1 so zero-degree nodes can still gain ties
    added = 0
    attempts = 0
    while added < n_add and attempts < n_add * 20:
        attempts += 1
        u = rng.choices(nodes, weights=weights, k=1)[0]
        v = rng.choices(nodes, weights=weights, k=1)[0]
        if u != v and not G2.has_edge(u, v):
            G2.add_edge(u, v)
            added += 1

    return G2


def generate_snapshot_sequence(n=100, m=3, T=15, seed=0):
    """Generate T evolving snapshots of a dynamic social network."""
    rng = random.Random(seed)
    G0 = make_initial_graph(n=n, m=m, seed=seed)
    snapshots = [G0]
    for t in range(1, T):
        snapshots.append(evolve_graph(snapshots[-1], rng=rng))
    return snapshots
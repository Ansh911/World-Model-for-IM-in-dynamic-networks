"""
Train the World Model on early snapshots of a dynamic social network, then
evaluate whether it can pick good seed sets on LATER, held-out snapshots --
faster than classical CELF, which must re-run expensive Monte Carlo
simulations from scratch every time the network changes.

THIS VERSION fixes three real problems found in earlier runs:

1. NON-DETERMINISM: PyTorch's multi-threaded CPU ops don't guarantee
   identical results run-to-run even with a fixed seed (floating-point
   addition order varies with thread scheduling). Forced single-threaded
   + deterministic algorithms so re-running with the same seed now
   actually reproduces the same trained model.

2. LOW MSE != GOOD RANKING: a model can have low average prediction
   error while still getting the RELATIVE ORDER of candidates wrong --
   which is all greedy seed selection actually depends on. Added a
   pairwise ranking loss (alongside the original MSE regression loss)
   that directly penalizes mis-ordered pairs of seed sets from the same
   snapshot, since that's the property that actually matters for
   planning quality.

3. NO SAFEGUARD AGAINST A BAD TRAINING RUN: previously just used
   whatever the model looked like after a fixed number of epochs, even
   if an earlier epoch was actually better. Added a genuine train/val/
   test split: the last training-range snapshot is held out as a
   validation snapshot (never directly trained on), and every few epochs
   we check the model's REAL seed-selection quality on it via actual
   simulation. The best-performing checkpoint by this real metric is
   restored before final evaluation on the true test snapshots -- so a
   noisy/unlucky final epoch can no longer wreck the reported result.

Also restructured the training loop to compute each snapshot's graph
embedding ONCE per epoch (not once per individual training example), and
combines spread + ranking + link losses into a single weighted loss per
snapshot per step, instead of firing off separate, competing gradient
updates -- this is both more correct (properly balanced) and much faster.

Usage:
    python train_world_model.py --real_data reddit_snapshots.pkl
    python train_world_model.py --real_data reddit_snapshots.pkl --celf_snapshots 4
    python train_world_model.py                      # synthetic fallback
"""

import argparse
import copy
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

torch.set_num_threads(1)  # eliminates multi-threaded floating-point nondeterminism
torch.use_deterministic_algorithms(True)

from dynamic_graph import generate_snapshot_sequence
from im_algorithms import build_propagation_probs, estimate_spread, celf
from worldmodel import (
    graph_to_sparse_norm_adj, node_features, GCNEncoder, SpreadPredictor,
    link_prediction_loss, world_model_greedy
)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--real_data", default=None)
arg_parser.add_argument("--node_names", default=None,
                         help="path to the *_node_names.pkl saved by process_reddit_hyperlinks.py, "
                              "so chosen seed sets get reported as real subreddit names instead of "
                              "anonymous integer IDs")
arg_parser.add_argument("--celf_snapshots", type=int, default=None)
arg_parser.add_argument("--r_label", type=int, default=50)
arg_parser.add_argument("--r_eval", type=int, default=300)
arg_parser.add_argument("--samples_per_snapshot", type=int, default=120)
arg_parser.add_argument("--epochs", type=int, default=100)
arg_parser.add_argument("--rank_loss_weight", type=float, default=1.5,
                         help="weight on the pairwise ranking loss, relative to spread MSE")
arg_parser.add_argument("--link_loss_weight", type=float, default=0.1,
                         help="weight on the link-prediction loss, kept low so it doesn't "
                              "destabilize the shared encoder")
arg_parser.add_argument("--max_rank_margin", type=float, default=20.0,
                         help="cap on the target margin for ranking pairs (uses the real "
                              "true-spread gap, capped, so big true gaps are pushed apart "
                              "more than tiny/noisy ones)")
arg_parser.add_argument("--val_every", type=int, default=10,
                         help="check real validation-snapshot quality every N epochs")
arg_parser.add_argument("--k", type=int, nargs="+", default=None,
                         help="seed set size(s) / budget(s), e.g. --k 10 20 30. The model is "
                              "trained once (using the largest k as the training budget) and "
                              "evaluated at every k given, so you get a results table across "
                              "budgets without retraining per budget. If omitted, auto-scales "
                              "a single value gently with graph size (8 at N=500, ~21 at "
                              "N=12,000, ~33 at N=55,863) so the budget stays meaningfully "
                              "proportioned as N grows, rather than a fixed k=8 becoming a "
                              "vanishingly small fraction on much bigger graphs.")
args = arg_parser.parse_args()

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if args.real_data:
    print(f"Loading REAL snapshots from {args.real_data} ...")
    with open(args.real_data, "rb") as f:
        snapshots = pickle.load(f)
    N = snapshots[0].number_of_nodes()
    T = len(snapshots)
else:
    print("No --real_data given, generating SYNTHETIC dynamic network snapshots...")
    N = 100
    T = 16
    snapshots = generate_snapshot_sequence(n=N, m=3, T=T, seed=SEED)

n_train = max(3, int(T * 0.7))
VAL_SNAPSHOT = n_train - 1               # held out from training, used for checkpoint selection
TRAIN_SNAPSHOTS = range(0, n_train - 1)  # actual training snapshots
TEST_SNAPSHOTS = range(n_train, T)       # true held-out final evaluation, untouched until the end

if args.k is not None:
    K_LIST = sorted(set(args.k))
else:
    K_LIST = [max(8, round(8 * (N / 500) ** 0.3))]  # 8 at N=500, ~21 at N=12,000, ~33 at N=55,863
K = max(K_LIST)  # K = largest budget requested; used to size training + checkpointing, unchanged
print(f"\nSeed budgets to evaluate: {K_LIST}  (training budget K = {K}, N = {N} nodes)")

for t, G in enumerate(snapshots):
    print(f"  t={t}: {N} nodes, {G.number_of_edges()} edges")

if args.celf_snapshots is not None:
    celf_snapshots = list(TEST_SNAPSHOTS)[:args.celf_snapshots]
elif N > 2000:
    celf_snapshots = list(TEST_SNAPSHOTS)[:1]
    print(f"\n[note] N={N} is large -- CELF's first round alone costs N*R_eval "
          f"= {N*args.r_eval:,} simulations. Capping CELF to 1 held-out snapshot "
          f"by default. Override with --celf_snapshots.")
else:
    celf_snapshots = list(TEST_SNAPSHOTS)

# ---------------------------------------------------------------------------
# Step 1: generate training labels, grouped by snapshot (needed for pairing)
# ---------------------------------------------------------------------------

print("\nGenerating training labels via real IC Monte Carlo simulation...")
rng = random.Random(SEED)
train_by_snapshot = defaultdict(list)  # t -> list of (seed_idx_tensor, true_spread)

t0 = time.time()
n_examples = 0
for t in TRAIN_SNAPSHOTS:
    G = snapshots[t]
    probs = build_propagation_probs(G)
    for _ in range(args.samples_per_snapshot):
        k = rng.randint(1, K)
        seeds = rng.sample(range(N), k)
        true_spread = estimate_spread(G, seeds, probs, args.r_label, rng)
        train_by_snapshot[t].append((torch.tensor(seeds, dtype=torch.long), true_spread))
        n_examples += 1
print(f"Generated {n_examples} labeled examples across {len(list(TRAIN_SNAPSHOTS))} training snapshots "
      f"in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Step 2: precompute SPARSE graph tensors for every snapshot
# ---------------------------------------------------------------------------

print("\nBuilding sparse graph tensors for each snapshot...")
t0 = time.time()
A_norms = [graph_to_sparse_norm_adj(G, N) for G in snapshots]
X_feats = [node_features(G, N) for G in snapshots]
print(f"Done in {time.time()-t0:.1f}s")


def pairwise_ranking_loss(predictor, Z, examples, rng, max_margin):
    """Penalizes pairs of seed sets (from the same snapshot, so directly
    comparable) that the model ranks in the wrong order. This is what
    greedy selection actually needs -- correct relative ordering, not
    correct absolute spread values."""
    if len(examples) < 2:
        return torch.tensor(0.0)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    losses = []
    for a, b in zip(idx[0::2], idx[1::2]):
        seed_i, true_i = examples[a]
        seed_j, true_j = examples[b]
        if true_i == true_j:
            continue
        pred_i = predictor(Z, seed_i)
        pred_j = predictor(Z, seed_j)
        target = 1.0 if true_i > true_j else -1.0
        margin = min(abs(true_i - true_j), max_margin)
        losses.append(F.relu(margin - target * (pred_i - pred_j)))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Step 3: train (combined loss per snapshot, deterministic, with checkpointing)
# ---------------------------------------------------------------------------

encoder = GCNEncoder(in_dim=3, hidden_dim=32, out_dim=16)
predictor = SpreadPredictor(emb_dim=16, hidden=32)
opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=0.01)

best_val_spread = -1.0
best_state = None
val_G = snapshots[VAL_SNAPSHOT]
val_probs = build_propagation_probs(val_G)

print(f"\nTraining world model for {args.epochs} epochs "
      f"(validation checkpoint check every {args.val_every} epochs)...")
t0 = time.time()
snapshot_list = list(TRAIN_SNAPSHOTS)

for epoch in range(args.epochs):
    random.shuffle(snapshot_list)
    total_spread, total_rank, total_link = 0.0, 0.0, 0.0

    for t in snapshot_list:
        Z = encoder(A_norms[t], X_feats[t])
        examples = train_by_snapshot[t]

        spread_terms = [(predictor(Z, seed_idx) - true_spread) ** 2 for seed_idx, true_spread in examples]
        spread_loss = torch.stack(spread_terms).mean()

        rank_loss = pairwise_ranking_loss(predictor, Z, examples, rng, args.max_rank_margin)

        if t + 1 < T:
            link_loss = link_prediction_loss(Z, snapshots[t + 1], N, rng=rng)
        else:
            link_loss = torch.tensor(0.0)

        combined = spread_loss + args.rank_loss_weight * rank_loss + args.link_loss_weight * link_loss

        opt.zero_grad()
        combined.backward()
        opt.step()

        total_spread += spread_loss.item()
        total_rank += rank_loss.item()
        total_link += link_loss.item()

    n_snap = len(snapshot_list)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:3d}: spread_MSE={total_spread/n_snap:7.2f}  "
              f"rank_loss={total_rank/n_snap:.3f}  ")

    if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
        val_seeds = world_model_greedy(encoder, predictor, A_norms[VAL_SNAPSHOT], X_feats[VAL_SNAPSHOT], N, K)
        val_spread = estimate_spread(val_G, val_seeds, val_probs, args.r_label, rng)
        if val_spread > best_val_spread:
            best_val_spread = val_spread
            best_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(predictor.state_dict()))

print(f"Training took {time.time()-t0:.1f}s")

if best_state is not None:
    encoder.load_state_dict(best_state[0])
    predictor.load_state_dict(best_state[1])
    print(f"\nRestored best checkpoint (validation spread = {best_val_spread:.1f}) for final evaluation")

# ---------------------------------------------------------------------------
# Step 4: evaluate on held-out FUTURE snapshots -- quality + speed
# ---------------------------------------------------------------------------

id_to_name = None
if args.node_names:
    with open(args.node_names, "rb") as f:
        id_to_name = pickle.load(f)
    print(f"\nLoaded node name mapping from {args.node_names} -- seed sets will be reported by name")


def describe_seeds(seed_list):
    if id_to_name is None:
        return str(seed_list)
    return str([id_to_name.get(s, f"node_{s}") for s in seed_list])


print("\n=== Evaluating on held-out future snapshots ===")
# results[k]["World Model" | "CELF (real simulation)"]["spread" | "time"] -> list
results = {k: {"World Model": {"spread": [], "time": []},
               "CELF (real simulation)": {"spread": [], "time": []}} for k in K_LIST}

for t in TEST_SNAPSHOTS:
    G = snapshots[t]
    probs = build_propagation_probs(G)

    # Plan ONCE per snapshot at the largest budget K, then take prefixes for
    # every smaller k in K_LIST. This is valid because both world_model_greedy
    # and celf build their seed set step-by-step, always adding whichever
    # single node looks best right now -- that choice never depends on how
    # many more nodes will be added later, so a run targeting K already
    # contains, as its first k nodes, exactly what a run targeting k directly
    # would have picked.
    t0 = time.time()
    wm_seeds_full = world_model_greedy(encoder, predictor, A_norms[t], X_feats[t], N, K)
    wm_plan_time = time.time() - t0

    if t in celf_snapshots:
        t0 = time.time()
        celf_seeds_full, _, _ = celf(G, K, probs, R=args.r_eval, seed=SEED)
        celf_time = time.time() - t0
    else:
        celf_seeds_full = None
        celf_time = None

    print(f"\nt={t}")
    for k in K_LIST:
        wm_seeds = wm_seeds_full[:k]
        wm_real_spread = estimate_spread(G, wm_seeds, probs, args.r_eval, rng)
        results[k]["World Model"]["spread"].append(wm_real_spread)
        results[k]["World Model"]["time"].append(wm_plan_time)
        print(f"  k={k:<4} World Model spread={wm_real_spread:.1f} (planned in {wm_plan_time:.3f}s)")
        print(f"          World Model seeds: {describe_seeds(wm_seeds)}")

        if celf_seeds_full is not None:
            celf_seeds = celf_seeds_full[:k]
            celf_real_spread = estimate_spread(G, celf_seeds, probs, args.r_eval, rng)
            results[k]["CELF (real simulation)"]["spread"].append(celf_real_spread)
            results[k]["CELF (real simulation)"]["time"].append(celf_time)
            print(f"          CELF spread={celf_real_spread:.1f} (planned in {celf_time:.2f}s)")
            print(f"          CELF seeds: {describe_seeds(celf_seeds)}")
        else:
            print("          CELF skipped (see --celf_snapshots)")
    print()
# ---------------------------------------------------------------------------
# Plot + summary
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ts = list(TEST_SNAPSHOTS)
celf_ts = celf_snapshots
colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(K_LIST), 2)))

for color, k in zip(colors, K_LIST):
    ax1.plot(ts, results[k]["World Model"]["spread"], "-o", color=color, label=f"World Model (k={k})")
    if results[k]["CELF (real simulation)"]["spread"]:
        ax1.plot(celf_ts, results[k]["CELF (real simulation)"]["spread"], "--s", color=color, alpha=0.6, label=f"CELF (k={k})")
ax1.set_xlabel("held-out snapshot (t)")
ax1.set_ylabel("actual spread achieved (real IC simulation)")
ax1.set_title("Seed quality on FUTURE, unseen network states")
ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

for color, k in zip(colors, K_LIST):
    ax2.plot(ts, results[k]["World Model"]["time"], "-o", color=color, label=f"World Model (k={k})")
    if results[k]["CELF (real simulation)"]["time"]:
        ax2.plot(celf_ts, results[k]["CELF (real simulation)"]["time"], "--s", color=color, alpha=0.6, label=f"CELF (k={k})")
ax2.set_yscale("log")
ax2.set_xlabel("held-out snapshot (t)")
ax2.set_ylabel("seed-selection time (seconds, log scale)")
ax2.set_title("Planning speed: learned model vs re-simulating")
ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("world_model_results.png", dpi=150)
print("\nSaved world_model_results.png")

print("\n=== SUMMARY (held-out future snapshots) ===")
for k in K_LIST:
    avg_wm_spread = np.mean(results[k]["World Model"]["spread"])
    avg_wm_time = np.mean(results[k]["World Model"]["time"])
    print(f"\n--- k={k} ---")
    print(f"World Model : avg spread = {avg_wm_spread:.1f}  | avg planning time = {avg_wm_time:.4f}s  (n={len(ts)} snapshots)")
    if results[k]["CELF (real simulation)"]["spread"]:
        avg_celf_spread = np.mean(results[k]["CELF (real simulation)"]["spread"])
        avg_celf_time = np.mean(results[k]["CELF (real simulation)"]["time"])
        print(f"CELF        : avg spread = {avg_celf_spread:.1f}  | avg planning time = {avg_celf_time:.4f}s  (n={len(celf_ts)} snapshots)")
        print(f"Quality retained: {100*avg_wm_spread/avg_celf_spread:.1f}% of CELF's spread")
        print(f"Speedup: {avg_celf_time/avg_wm_time:.1f}x faster planning")
    else:
        print("CELF: skipped on all snapshots -- no comparison available this run.")
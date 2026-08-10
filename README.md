# World Model for Influence Maximization in Dynamic Networks

A graph-based World Model for fast influence maximization in **dynamic social networks**.

The project learns a compact representation of evolving graph snapshots and uses the learned model to select influential seed nodes without repeatedly running expensive influence-spread simulations during seed selection.

The main goal is to achieve influence spread comparable to CELF while dramatically reducing seed-selection time.

# Overview

Influence Maximization (IM) aims to identify a small set of seed nodes that can maximize information propagation through a network.

Traditional greedy methods such as **CELF** repeatedly estimate the marginal influence of candidate nodes using Independent Cascade (IC) simulations. While effective, this becomes extremely expensive for large dynamic networks.

This project proposes a learned **World Model** that:

1. Represents each temporal graph snapshot using a GCN-based encoder.
2. Learns node and global graph representations.
3. Predicts influence spread for candidate seed sets.
4. Uses the learned predictor for greedy seed selection.
5. Evaluates the selected seeds using the actual Independent Cascade model.

This allows seed selection to be performed much faster than simulation-based CELF.

# Method

The input network is represented as a sequence of temporal snapshots:

G₀ → G₁ → G₂ → ... → Gₜ

#Running the Experiments
1. Prepare the temporal dataset

The model expects a preprocessed temporal snapshot file:

reddit_snapshots.pkl

or the corresponding .pkl file generated for another SNAP temporal network.

ex: for AskUbuntu

python3 preprocess_snap.py --input sx-askubuntu.txt.gz --output askubuntu_snapshots.pkl

2. Run the World Model

Basic experiment with seed budgets K = 10, 20, 30:

python3 train_wm.py \
    --real_data askubuntu_snapshots.pkl \
    --celf_snapshots 4 \
    --samples_per_snapshot 200 \
    --k 10 20 30



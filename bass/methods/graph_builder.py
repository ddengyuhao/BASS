import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Visual vocabulary (vector quantization of local patch features)
# ---------------------------------------------------------------------------

def build_visual_vocabulary(patches, vocab_size=1024, n_iter=20, sample_size=100000,
                            device="cpu", seed=0):
    """
    Spherical k-means over patch embeddings.

    Args:
        patches: (M, D) patch features (any dtype, any device).
        vocab_size: number of visual words |W|.
        n_iter: k-means iterations.
        sample_size: number of patches used to fit the codebook.
    Returns:
        codebook: (K, D) L2-normalized centroids on `device`.
    """
    gen = torch.Generator().manual_seed(seed)
    M = patches.shape[0]
    if M > sample_size:
        idx = torch.randperm(M, generator=gen)[:sample_size]
        x = patches[idx]
    else:
        x = patches
    x = F.normalize(x.to(device=device, dtype=torch.float32), dim=-1)

    K = min(vocab_size, x.shape[0])
    centers = x[torch.randperm(x.shape[0], generator=gen)[:K].to(device)].clone()

    for _ in range(n_iter):
        assign = _nearest_center(x, centers)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=K)

        # Re-seed empty clusters with random points
        empty = counts == 0
        if empty.any():
            refill = torch.randint(0, x.shape[0], (int(empty.sum()),), generator=gen).to(device)
            new_centers[empty] = x[refill]
        centers = F.normalize(new_centers, dim=-1)

    return centers


def quantize_patches(patches, codebook, chunk_size=65536):
    """Maps each patch to the id of its nearest visual word."""
    device = codebook.device
    out = []
    for i in range(0, patches.shape[0], chunk_size):
        x = F.normalize(patches[i:i + chunk_size].to(device=device, dtype=torch.float32), dim=-1)
        out.append(_nearest_center(x, codebook).cpu())
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long)


def _nearest_center(x, centers, chunk_size=65536):
    out = []
    for i in range(0, x.shape[0], chunk_size):
        out.append(torch.mm(x[i:i + chunk_size], centers.t()).argmax(dim=1))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Semantic edges via inverted index + sparse similarity join (Sec. 3)
# ---------------------------------------------------------------------------

def build_inverted_index(event_words, num_events):
    """
    Args:
        event_words: list of 1-D LongTensors, the visual word ids of each event.
    Returns:
        postings: dict word_id -> np.ndarray of event ids containing the word
                  (each word counted once per event).
    """
    postings = {}
    for i, words in enumerate(event_words):
        for z in torch.unique(words).tolist():
            postings.setdefault(z, []).append(i)
    return {z: np.asarray(ev, dtype=np.int64) for z, ev in postings.items()}


def compute_semantic_edges(global_feats, event_words, delta=0.65, rho=50, uniform_weight=False):
    """
    Constructs forward semantic edges v_i -> v_j (i < j, non-adjacent) with
        s_ij = 0.5 * (h_i^T h_j + cos(x_i^g, x_j^g))            (Eq. 5),
    where h_i is the L2-normalized vector of inverse-event-frequency weights
    gamma_z = log(N / N_z) (Eq. 4).
    Only event pairs that share at least one retained visual word are scored.

    Args:
        global_feats: (N, D) event-level global features.
        event_words: list of N LongTensors of visual word ids.
        delta: semantic similarity threshold.
        rho: posting-list cap; words occurring in more than rho events are dropped.
        uniform_weight: use weight 1 for every retained visual word instead of the
            inverse-event-frequency weight (the "w/o Weighting" variant in Table 3).
    Returns:
        edges: dict (i, j) -> s_ij for all retained semantic edges.
        stats: dict with posting-list visit count and number of candidate pairs.
    """
    N = len(event_words)
    postings = build_inverted_index(event_words, N)

    # Inverse-event-frequency weights; stop words (N_z > rho) are discarded
    gamma = {}
    for z, ev in postings.items():
        n_z = len(ev)
        if n_z <= rho:
            gamma[z] = 1.0 if uniform_weight else math.log(N / n_z)

    # L2 norm of each sparse event vector h_i
    sq_norm = np.zeros(N, dtype=np.float64)
    for z, g in gamma.items():
        sq_norm[postings[z]] += g * g
    norm = np.sqrt(sq_norm)
    norm[norm == 0] = 1.0

    # Sparse join: accumulate h_i^T h_j by walking each retained posting list
    sim_w = np.zeros((N, N), dtype=np.float64)
    visits = 0
    for z, g in gamma.items():
        ev = postings[z]
        if len(ev) < 2 or g == 0:
            continue
        w = g / norm[ev]
        sim_w[np.ix_(ev, ev)] += np.outer(w, w)
        visits += len(ev) * len(ev)

    g_norm = F.normalize(global_feats.float(), dim=-1)
    sim_g = torch.mm(g_norm, g_norm.t()).cpu().numpy()

    edges = {}
    cand_i, cand_j = np.nonzero(np.triu(sim_w, k=2))  # i < j and non-adjacent
    for i, j in zip(cand_i.tolist(), cand_j.tolist()):
        s_ij = 0.5 * (sim_w[i, j] + sim_g[i, j])
        if s_ij >= delta:
            edges[(i, j)] = float(s_ij)

    stats = {"posting_visits": visits, "candidate_pairs": len(cand_i)}
    return edges, stats


def compute_semantic_edges_pairwise(global_feats, event_patches, delta=0.65, chunk_size=4096):
    """
    Dense pairwise matching baseline ("PM" in Table 3): for every non-adjacent pair
    i < j, computes the patch-level score of Eq. (3),
        s_hat_ij = 1/L_i * sum_k max_l cos(p_ik, p_jl),
    without vector quantization, and s_ij = 0.5 * (s_hat_ij + cos(x_i^g, x_j^g)).
    """
    N = len(event_patches)
    device = global_feats.device
    patches = [F.normalize(p.to(device=device, dtype=torch.float32), dim=-1) for p in event_patches]

    g_norm = F.normalize(global_feats.float(), dim=-1)
    sim_g = torch.mm(g_norm, g_norm.t()).cpu()

    edges = {}
    for i in range(N):
        for j in range(i + 2, N):
            best = torch.full((patches[i].shape[0],), -1.0, device=device)
            for c in range(0, patches[j].shape[0], chunk_size):
                best = torch.maximum(best, torch.mm(patches[i], patches[j][c:c + chunk_size].t()).max(dim=1).values)
            s_ij = 0.5 * (float(best.mean()) + float(sim_g[i, j]))
            if s_ij >= delta:
                edges[(i, j)] = s_ij
    return edges, {"pairs": N * (N - 1) // 2}


def build_adjacency(num_events, semantic_edges):
    """
    Weighted adjacency of the event graph: temporal edges v_i -> v_{i+1} (Eq. 6)
    with weight 1.0 plus the semantic edges.
    """
    A = torch.zeros((num_events, num_events), dtype=torch.float32)
    idx = torch.arange(num_events - 1)
    A[idx, idx + 1] = 1.0
    for (i, j), w in semantic_edges.items():
        A[i, j] = w
    return A


# ---------------------------------------------------------------------------
# Reachability matrix (Personalized PageRank, Sec. 4.1)
# ---------------------------------------------------------------------------

def compute_pagerank_matrix(adj, alpha=0.15, tol=1e-6, max_iter=200):
    """
    Solves Pi = alpha * I + (1 - alpha) * Pi P (Eq. 7) by power iteration, where
    P = D^{-1} A is the row-normalized transition matrix. Nodes with zero
    out-degree receive a self-loop before normalization.

    Row v of Pi is the PPR distribution of a random walk restarting at v.
    """
    N = adj.shape[0]
    device = adj.device
    A = adj.clone()

    out_deg = A.sum(dim=1)
    sinks = out_deg == 0
    if sinks.any():
        A[sinks, sinks] = 1.0
    P = A / A.sum(dim=1, keepdim=True)

    eye = torch.eye(N, device=device)
    Pi = alpha * eye
    for _ in range(max_iter):
        Pi_next = alpha * eye + (1 - alpha) * torch.mm(Pi, P)
        if torch.max(torch.abs(Pi_next - Pi)) < tol:
            Pi = Pi_next
            break
        Pi = Pi_next
    return Pi

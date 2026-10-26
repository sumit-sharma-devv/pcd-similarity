"""Point cloud similarity metrics.

Primary metric: Density-aware Chamfer Distance (DCD), Wu et al., NeurIPS 2021
(arXiv:2111.12702):

    DCD(S1, S2) = 1/2 * ( 1/|S1| * sum_{x in S1} [1 - 1/n_y * exp(-alpha * ||x - y||_2)]
                        + 1/|S2| * sum_{y in S2} [1 - 1/n_x * exp(-alpha * ||y - x||_2)] )

where y is the nearest neighbour of x in the other set and n_y is the number of
points of the querying set that pick y as their nearest neighbour (this penalises
density mismatch). DCD lies in [0, 1]; we report similarity = 1 - DCD.
"""
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_ALPHA = 1000.0  # paper default
DEFAULT_TAU = 0.01  # F-score threshold, relative to the unit-scale cloud


def normalize_pair(a: np.ndarray, b: np.ndarray):
    """Centre on the first cloud and scale both by its bounding-sphere radius.

    The reference cloud `a` defines the frame, so a global size/offset difference
    between the two scans is still penalised (as the paper evaluates in a shared frame).
    """
    centre = a.mean(axis=0)
    scale = np.linalg.norm(a - centre, axis=1).max()
    scale = scale if scale > 0 else 1.0
    return (a - centre) / scale, (b - centre) / scale


def _directional(src, dst_tree, n_dst):
    d, idx = dst_tree.query(src, workers=-1)
    counts = np.bincount(idx, minlength=n_dst)
    return d, idx, counts[idx]


def compute_metrics(a: np.ndarray, b: np.ndarray, alpha=DEFAULT_ALPHA, tau=DEFAULT_TAU,
                    normalize=True, max_points=200_000, seed=0):
    sampled = False
    rng = np.random.default_rng(seed)
    if len(a) > max_points:
        a, sampled = a[rng.choice(len(a), max_points, replace=False)], True
    if len(b) > max_points:
        b, sampled = b[rng.choice(len(b), max_points, replace=False)], True
    if normalize:
        a, b = normalize_pair(a, b)

    ta, tb = cKDTree(a), cKDTree(b)
    d_ab, idx_ab, n_ab = _directional(a, tb, len(b))  # each point of a -> nearest in b
    d_ba, idx_ba, n_ba = _directional(b, ta, len(a))

    # n_y: how many querying points share the same nearest neighbour y.
    dcd = 0.5 * (np.mean(1 - np.exp(-alpha * d_ab) / n_ab) +
                 np.mean(1 - np.exp(-alpha * d_ba) / n_ba))

    cd_l1 = 0.5 * (d_ab.mean() + d_ba.mean())
    cd_l2 = 0.5 * ((d_ab ** 2).mean() + (d_ba ** 2).mean())
    hausdorff = max(d_ab.max(), d_ba.max())
    precision, recall = (d_ba < tau).mean(), (d_ab < tau).mean()
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "similarity": float(1 - dcd),
        "dcd": float(dcd),
        "chamfer_l1": float(cd_l1),
        "chamfer_l2": float(cd_l2),
        "hausdorff": float(hausdorff),
        "fscore": float(fscore),
        "points_a": int(len(a)),
        "points_b": int(len(b)),
        "alpha": alpha,
        "tau": tau,
        "normalized": normalize,
        "subsampled": sampled,
    }

import random
import numpy as np
from collections import defaultdict


def get_support_set(train_ds, strategy="first", k_shots=1, seed=None):
    """
    Selects examples from the training dataset to form the ICL support set.

    Args:
        train_ds:  Dataset that yields (time_series, label) pairs.
        strategy:  One of "first" | "random" | "medoid" | "medoid_dtw".
                   - "first"      : take the first k examples per class.
                   - "random"     : sample k examples per class uniformly at random.
                   - "medoid"     : pick the k examples per class whose sum of
                                    L2 distances to all other class members is smallest.
                   - "medoid_dtw" : same as medoid but uses DTW distance.
        k_shots:   Number of examples to select per class.
        seed:      Optional random seed (used by "random" strategy).

    Returns:
        List of (ts_list, label) tuples.
    """
    if strategy == "first":
        return _first(train_ds, k_shots)
    elif strategy == "random":
        return _random(train_ds, k_shots, seed)
    elif strategy == "medoid":
        return _medoid(train_ds, k_shots)
    elif strategy == "medoid_dtw":
        return _medoid_dtw(train_ds, k_shots)
    elif strategy == "reversed":
        return _reversed(train_ds, k_shots)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: first, random, medoid, medoid_dtw, reversed.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_array(ts):
    if hasattr(ts, "numpy"):
        return ts.numpy().flatten().astype(float)
    if hasattr(ts, "tolist"):
        return np.array(ts).flatten().astype(float)
    return np.array(ts).flatten().astype(float)


def _to_list(ts):
    if hasattr(ts, "tolist"):
        return [ts.tolist()]
    return [ts]


def _label_val(label):
    if hasattr(label, "item"):
        return label.item()
    return label


# ── strategies ────────────────────────────────────────────────────────────────

def _first(train_ds, k_shots):
    examples = []
    label_counts = {}
    for i in range(len(train_ds)):
        ts_i, label_i = train_ds[i]
        label = _label_val(label_i)
        if label_counts.get(label, 0) < k_shots:
            examples.append((_to_list(ts_i), label))
            label_counts[label] = label_counts.get(label, 0) + 1
    return examples


def _reversed(train_ds, k_shots):
    return _first(train_ds, k_shots)[::-1]


def _random(train_ds, k_shots, seed=None):
    rng = random.Random(seed)

    # Group all indices by label
    by_label = defaultdict(list)
    for i in range(len(train_ds)):
        _, label_i = train_ds[i]
        by_label[_label_val(label_i)].append(i)

    examples = []
    for label, indices in by_label.items():
        chosen = rng.sample(indices, min(k_shots, len(indices)))
        for idx in chosen:
            ts_i, _ = train_ds[idx]
            examples.append((_to_list(ts_i), label))
    return examples


def _dtw_distance(a, b):
    """Sakoe-Chiba DTW with no window constraint."""
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return np.sqrt(dtw[n, m])


def _medoid_dtw(train_ds, k_shots):
    by_label = defaultdict(list)
    for i in range(len(train_ds)):
        ts_i, label_i = train_ds[i]
        by_label[_label_val(label_i)].append((i, _to_array(ts_i)))

    examples = []
    for label, items in by_label.items():
        if len(items) == 1:
            idx, _ = items[0]
            ts_i, _ = train_ds[idx]
            examples.append((_to_list(ts_i), label))
            continue

        # Sum of DTW distances from each item to every other item in the class
        dist_sums = np.zeros(len(items))
        for j in range(len(items)):
            for k in range(len(items)):
                if k != j:
                    dist_sums[j] += _dtw_distance(items[j][1], items[k][1])

        best = np.argsort(dist_sums)[:k_shots]
        for b in best:
            orig_idx, _ = items[b]
            ts_i, _ = train_ds[orig_idx]
            examples.append((_to_list(ts_i), label))

    return examples


def _medoid(train_ds, k_shots):
    # Group all (array, index) by label
    by_label = defaultdict(list)
    for i in range(len(train_ds)):
        ts_i, label_i = train_ds[i]
        by_label[_label_val(label_i)].append((i, _to_array(ts_i)))

    examples = []
    for label, items in by_label.items():
        if len(items) == 1:
            idx, ts_arr = items[0]
            ts_i, _ = train_ds[idx]
            examples.append((_to_list(ts_i), label))
            continue

        arrays = np.stack([arr for _, arr in items])   # (N, T)

        # Sum of L2 distances from each item to every other item in the class
        dist_sums = np.zeros(len(items))
        for j in range(len(items)):
            diffs = arrays - arrays[j]                 # (N, T)
            dist_sums[j] = np.sqrt((diffs ** 2).sum(axis=1)).sum()

        # Pick k indices with smallest distance sum
        best = np.argsort(dist_sums)[:k_shots]
        for b in best:
            orig_idx, _ = items[b]
            ts_i, _ = train_ds[orig_idx]
            examples.append((_to_list(ts_i), label))

    return examples

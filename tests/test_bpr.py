import itertools
from collections import defaultdict
from functools import partial

import torch

from motive.bpr import create_positive_negative_pairs

assert_equal = partial(
    torch.testing.assert_close, rtol=0, atol=0, check_dtype=False, check_device=False
)


def _lexsort(indices):
    if indices.numel() == 0:
        return None
    max_val = indices.max() + 1
    combined = indices[:, 0] * max_val + indices[:, 1]
    return torch.argsort(combined)


def create_positive_negative_pairs_naive(indices, labels):
    indices = indices.cpu().numpy()
    labels = labels.cpu().numpy()
    elems = defaultdict(lambda: ([], []))
    for i, (id, label) in enumerate(zip(indices, labels)):
        elems[id][int(label)].append(i)

    pairs = []
    weights = []
    for id, (neg, pos) in elems.items():
        prod = list(itertools.product(pos, neg))
        if prod:
            pairs.extend(prod)
            weights.extend([1.0 / len(prod)] * len(prod))
    pairs = torch.tensor(pairs, dtype=torch.int64).reshape([-1, 2])
    weights = torch.tensor(weights, dtype=torch.float32)
    return pairs, weights


def test_find_pairs():
    torch.manual_seed(0)
    for _ in range(1000):
        # Generate random inputs
        num_indices = torch.randint(100, 200, (1,))
        indices = torch.randint(100, 1000, (num_indices,))
        labels = torch.randint(0, 2, (num_indices,))

        # Calculate outputs from both functions
        pairs_naive, weights_naive = create_positive_negative_pairs_naive(
            indices, labels
        )
        idx_naive = _lexsort(pairs_naive)
        pairs_naive = pairs_naive[idx_naive]
        weights_naive = weights_naive[idx_naive]

        pairs, weights = create_positive_negative_pairs(indices, labels)
        idx = _lexsort(pairs)
        pairs = pairs[idx]
        weights = weights[idx]

        # Assert that the outputs are equal
        assert_equal(pairs_naive.squeeze(), pairs.squeeze())
        assert_equal(weights_naive.squeeze(), weights.squeeze())
        print(weights)
        assert len(pairs) == len(weights)

import torch


def create_positive_negative_pairs(
    indices: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Create all possible pairs between positive and negative samples for each unique index.

    This function generates pairs of indices where one index corresponds to a positive
    sample and the other to a negative sample, for each unique identifier in the input.

    Parameters:
    -----------
    indices : torch.Tensor
        A 1D tensor of integer indices, where each index represents an identifier.
    labels : torch.Tensor
        A 1D tensor of boolean labels corresponding to the indices. True represents
        a positive sample, False represents a negative sample.

    Returns:
    --------
    torch.Tensor
        A 2D tensor of shape (T, 2), where T is the total number of pairs.
        Each row contains two indices: [positive_index, negative_index].

    Example:
    --------
    >>> indices = torch.tensor([0, 0, 1, 1, 2, 2])
    >>> labels = torch.tensor([True, False, True, False, True, False])
    >>> pairs, weights = create_positive_negative_pairs(indices, labels)
    >>> print(pairs)
    tensor([[0, 1],
            [2, 3],
            [4, 5]])

    Notes:
    ------
    - The function assumes that `indices` and `labels` have the same length.
    - It uses boolean operations and tensor manipulations for efficiency.
    - The resulting pairs are unique combinations for each identifier.
    """
    # n: number of samples, u: number of unique ids
    uniq = torch.unique(indices)  # (u, )
    id_mask = uniq[:, None] == indices  # (u, n) = (u, 1) == (n,)
    labels = labels.bool()  # (n, )

    pos_mask = id_mask * labels  # (u, n) = (u, n) * (n,)
    neg_mask = id_mask * ~labels  # (u, n) = (u, n) * (n,)
    # P tuples (id, ix) locating the positives for all id \in uniq
    pos_ix = pos_mask.nonzero()
    # N tuples (id, ix) locating the negatives for all id \in uniq
    neg_ix = neg_mask.nonzero()

    counts = pos_mask.sum(axis=1) * neg_mask.sum(axis=1)
    weights = torch.repeat_interleave(1 / counts, counts)

    mask = pos_ix[:, 0][:, None] == neg_ix[:, 0]  # (P, N) = (P, 1) == (N, )
    pairs = mask.nonzero()  # (T, 2) cartesian product of pos and neg for each id
    pos_ix = pos_ix[pairs[:, 0], 1]
    neg_ix = neg_ix[pairs[:, 1], 1]
    result = torch.stack([pos_ix, neg_ix], dim=1)
    return result, weights

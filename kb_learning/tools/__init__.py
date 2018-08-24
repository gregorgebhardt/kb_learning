import itertools
import numpy as np


def compute_robust_mean_swarm_position(state, percentage=.8):
    # compute distances to all other positions and sum them up for each kilobot, find percentage with smallest
    # distances to rest of the swarm. Compute mean over these positions
    # transform state
    from scipy.spatial.distance import pdist, squareform

    state = state.reshape(-1, 2)
    # number of kilobots to include in the mean
    num_included = int(np.ceil(state.shape[0] * percentage))
    # compute pairwise distances and sum over fist axis, i.e. sum of distances for each position
    distances = squareform(pdist(state)).sum(axis=0)
    # find an index that partitions the distances such
    index = np.argpartition(distances, num_included)

    return np.mean(state[index[:num_included], :], axis=0)


def chunks(iterable, n):
    it = iter(iterable)

    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def np_chunks(array: np.ndarray, n):
    for chunk in chunks(array, n):
        yield np.array(chunk)

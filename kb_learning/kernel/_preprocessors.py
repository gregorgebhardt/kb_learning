import numpy as np
import pandas as pd
import scipy.spatial as spatial


def compute_median_bandwidth(data, quantile=.5, sample_size=1000):
    """Computes a bandwidth for the given data set using the median heuristic.
    Other quantiles can be chosen with the quantile keyword argument.

    Arguments:
    data -- a DataFrame with the variables in the columns
    quantile -- scalar or list of scalars from the range (0,1)
    sample_size -- maximum number of sample to compute the point-wise distances

    Returns:
    bandwidths -- an array with the bandwidth for each variable
    """
    if len(data.shape) > 1:
        num_variables = data.shape[1]
    else:
        num_variables = 1

    bandwidths = np.zeros(num_variables)

    num_data_points = data.shape[0]

    if sample_size > num_data_points:
        data_points = data.values
    else:
        data_points = data.sample(sample_size).values

    for i in range(num_variables):
        distances = spatial.distance.pdist(data_points[:, i:i+1])
        if quantile == .5:
            bandwidths[i] = np.median(distances)
        else:
            bandwidths[i] = pd.DataFrame(distances).quantile(quantile)

    return bandwidths


def select_reference_set_randomly(data, size, consecutive_sets=1, group_by=None):
    """selects a random reference set from the given DataFrame. Consecutive sets are computed from the first random
    reference set, where it is assured that only data points are chosen for the random set that have the required
    number of successive data points. Using the group_by argument allows to ensure that all consecutive samples are
    from the same group.

    :param data: a pandas.DataFrame with the samples to choose from
    :param size: the number of samples in the reference set
    :param consecutive_sets: the number of consecutive sets returned by this function (default: 1)
    :param group_by: a group_by argument to ensure that the consecutive samples are from the same group as the first
    random sample
    :return: a tuple with the reference sets
    """
    weights = np.ones(data.shape[0])

    if group_by is not None:
        gb = data.groupby(level=group_by)
        last_windows_idx = [ix[-i] for _, ix in gb.indices.items() for i in range(1, consecutive_sets)]
        weights[last_windows_idx] = 0
    else:
        last_windows_idx = [data.index[-i] for i in range(1, consecutive_sets + 1)]
        weights[last_windows_idx] = 0

    # select reference set
    if weights.sum() <= size:
        # if there is not enough data, we take all data points
        reference_set1 = data.loc[weights == 1].index.sort_values()
    else:
        # otherwise we chose a random reference set from the data
        reference_set1 = data.sample(n=size, weights=weights).index.sort_values()

    if consecutive_sets > 1:
        reference_set = [reference_set1]
        for i in range(1, consecutive_sets):
            if type(reference_set1) is pd.MultiIndex:
                reference_set_i = pd.MultiIndex.from_tuples([*map(lambda t: (*t[:-1], t[-1] + i),
                                                                  reference_set1.values)])
                reference_set_i.set_names(reference_set1.names, inplace=True)
                reference_set.append(reference_set_i)
            else:
                reference_set_i = pd.Index(data=reference_set1 + i, name=reference_set1.name)
                reference_set.append(reference_set_i)

    else:
        reference_set = reference_set1

    return tuple(reference_set)

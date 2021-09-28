import numpy as np


def _merge_intervals_inplace(merge_target, merger, sum_weighted_y, sum_weighted_y_sq, sum_weights, level_set):
    sum_weighted_y[merge_target] += sum_weighted_y[merger]
    sum_weighted_y_sq[merge_target] += sum_weighted_y_sq[merger]
    sum_weights[merge_target] += sum_weights[merger]

    # Update the level set
    level_set[merge_target] = sum_weighted_y[merge_target] / sum_weights[merge_target]


def prefix_isotonic_regression(y, weights=None, non_negativity=False):
    if weights is None:
        weights = np.ones_like(y)

    sumwy = weights * y
    sumwy2 = weights * y * y
    sumw = weights.copy()

    level_set = np.zeros_like(y)
    index_range = np.zeros_like(y, dtype=np.int32)
    error = np.zeros(y.shape[0] + 1)  # +1 since error[0] is error of empty set

    level_set[0] = y[0]
    index_range[0] = 0
    num_samples = y.shape[0]

    if non_negativity:
        cumsumwy2 = np.cumsum(sumwy2)
        threshold = np.zeros(level_set.shape)
        if level_set[0] < 0:
            threshold[0] = True
            error[1] = cumsumwy2[0]

    for i in range(1, num_samples):
        level_set[i] = y[i]
        index_range[i] = i
        while level_set[i] <= level_set[index_range[i]-1] and index_range[i] != 0:
            _merge_intervals_inplace(i, index_range[i]-1, sumwy, sumwy2, sumw, level_set)
            index_range[i] = index_range[index_range[i] - 1]

        levelerror = sumwy2[i] - (sumwy[i]**2 / sumw[i])
        if non_negativity and level_set[i] < 0:
            threshold[i] = True
            error[i + 1] = cumsumwy2[i]
        else:
            error[i + 1] = levelerror + error[index_range[i]]

    if non_negativity:
        for i in range(len(level_set)):
            if threshold[i]:
                level_set[i] = 0
    

    error = np.zeros(y.shape[0] + 1)  # +1 since error[0] is error of empty set
    for i in range(1, y.shape[0]+1):
        yhat = compute_isotonic_from_index(i, level_set, index_range)
        error[i] = np.sum((yhat - y[:i])**2)

    return (level_set, index_range), error


def compute_isotonic_from_index(end_index, level_set, index_range):
    if end_index is None:
        idx = level_set.shape[0] - 1
    else:
        idx = end_index - 1

    y_iso = np.empty_like(level_set[:idx+1]) * np.nan

    while idx >= 0:
        y_iso[index_range[idx]:idx+1] = level_set[idx]
        idx = index_range[idx] - 1
    assert not np.any(np.isnan(y_iso))

    return y_iso


def _get_best_unimodality_index(error_left, error_right):
    best_error = error_right[-1]
    best_idx = 0
    for i in range(error_left.shape[0]):
        error = error_left[i] + error_right[len(error_left) - i - 1]
        if error < best_error:
            best_error = error
            best_idx = i
    return best_idx, best_error


def _unimodal_regression(y, non_negativity):
    iso_left, error_left = prefix_isotonic_regression(y, non_negativity=non_negativity)
    iso_right, error_right = prefix_isotonic_regression(y[::-1], non_negativity=non_negativity)


    num_samples = y.shape[0]
    best_idx, error = _get_best_unimodality_index(error_left, error_right)
    y_iso_left = compute_isotonic_from_index(best_idx, iso_left[0], iso_left[1])
    y_iso_right = compute_isotonic_from_index(num_samples-best_idx, iso_right[0], iso_right[1])

    return np.concatenate([y_iso_left, y_iso_right[::-1]]), error


def unimodal_regression(y, non_negativity=False):
    y = np.asarray(y)
    if y.ndim == 1:
        return _unimodal_regression(y, non_negativity=non_negativity)[0]
    elif y.ndim == 2:
        return np.stack([_unimodal_regression(y[:, r], non_negativity=non_negativity)[0] for r in range(y.shape[1])], axis=1)
    else:
        raise ValueError(f"y must be a vector or matrix, has {y.ndim} dimensions.")
     

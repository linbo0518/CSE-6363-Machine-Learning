import numpy as np


def distance(x1, x2):
    dist = 0
    for i1, i2 in zip(x1, x2):
        try:
            dist += (int(i1) - int(i2))**2
        except ValueError:
            dist += _compare_str(i1, i2)
    return dist


def _compare_str(str1, str2):
    dist = 0
    for c1, c2 in zip(str1, str2):
        if (c1 != c2):
            dist += 1
    dist += abs(len(str1) - len(str2))
    return dist


def convert(x):
    n_rows, n_cols = x.shape
    new_cols = 0
    flag_cols = [0]
    for col_idx in range(n_cols):
        try:
            x[:, col_idx].astype(np.float)
            new_cols += 1
        except ValueError:
            new_cols += len(np.unique(x[:, col_idx]))
        flag_cols.append(new_cols)
    one_hot = np.zeros((n_rows, new_cols))
    for col_idx, start_idx, end_idx in zip(range(n_cols), flag_cols[:-1],
                                           flag_cols[1:]):
        try:
            col = x[:, col_idx].astype(np.float)
            min_value = np.min(col)
            max_value = np.max(col)
            one_hot[:, start_idx:end_idx] = np.expand_dims(
                (col - min_value) / (max_value - min_value), axis=1)
        except ValueError:
            one_hot[:, start_idx:end_idx] = _one_hot(x[:, col_idx])
    return one_hot


def _one_hot(col):
    unique, col_idx = np.unique(col, return_inverse=True)
    one_hot = np.zeros((len(col), len(unique)))
    for r, c in zip(np.arange(len(col)), col_idx):
        one_hot[r, c] = 1
    return one_hot

import numpy as np
import itertools

def custom_dist(pt1, pt2, metric):
    if metric == 'e':
        dist = np.linalg.norm(pt1-pt2)
    if metric == 's':
        d1, cc = st.pearsonr(pt1, pt2)
        dist = 1-d1
    return dist


def dmatrix(data, metric='e'):
    # does linkage among columns
    if metric == 's':
        data_s = data.rank()
        data_s = np.array(data_s)
    else:
        data_s = np.array(data)
    tuples = list(itertools.combinations(range(data_s.shape[1]), 2))
    vec = np.zeros(len(tuples))
    for k, (i, j) in enumerate(tuples):
        if i < j:
            vec[k] = custom_dist(data_s[:, i], data_s[:, j], metric)
    return vec


def isclose(a, b, tol=1e-03):
    return (abs(a-b) <= tol).all()

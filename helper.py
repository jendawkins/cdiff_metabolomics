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


def leave_one_out_cv(data, labels, num_folds=None):

    if isinstance(data.index.values[0], str):
        patients = np.array([int(i.split('-')[1])
                                for i in data.index.values])
        pdict = {}
        for i, pt in enumerate(patients):
            pdict[pt] = labels[i]

        ix_all = []
        for ii in pdict.keys():
            pt_test = ii
            pt_train = list(set(pdict.keys()) - set([ii]))
            ixtrain = (np.concatenate(
                [np.where(patients == j)[0] for j in pt_train]))
            ixtest = np.where(patients == pt_test)[0]
            set1 = set([patients[ix] for ix in ixtest])
            set2 = set([patients[ix] for ix in ixtrain])
            set1.intersection(set2)

            ix_all.append((ixtrain, ixtest))
            assert(not set1.intersection(set2))

    else:
        ix_all = []
        # CHANGE LINE!
        for ixs in range(len(labels)):
            ixtest = [ixs]
            ixtrain = list(set(range(len(labels))) - set(ixtest))
            ix_all.append((ixtrain, ixtest))
    return ix_all


def isclose(a, b, tol=1e-03):
    return (abs(a-b) <= tol).all()

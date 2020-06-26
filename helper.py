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


# # def make_summary_fig():
#     import matplotlib as mpl

#     plt.rcParams.update({'font.size': 25})

#     img = np.zeros((len(cd.pt_info_dict.keys()), len(np.arange(0, 8.5, .5))))
#     times = np.arange(0, 8.5, .5)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if times[j] in ml.new_info_dict[i].keys():
#                 if ml.new_info_dict[i][times[j]]['PATIENT STATUS (BWH)'] == 'Recur':
#                     img[i, j] = 2
#                 else:
#                     img[i, j] = 1


#     # Create Cmap
#     cmap = plt.cm.jet  # define the colormap
#     # extract all colors from the .jet map
#     cmaplist = [cmap(i) for i in range(cmap.N)]
#     # force the first color entry to be grey
#     cmaplist[0] = (.5, .5, .5, 1.0)
#     cmaplist[-1] = (1, 0, 0, 1.0)
#     # create the new map
#     cmap = mpl.colors.LinearSegmentedColormap.from_list(
#         'Custom cmap', cmaplist, cmap.N)

#     # define the bins and normalize
#     bounds = np.linspace(0, 3, 4)
#     norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#     fig, ax = plt.subplots(figsize=(img.shape[0], np.round(img.shape[1]*1.5)))
#     im = ax.imshow(img, cmap=cmap)

#     # create a second axes for the colorbar
#     ax2 = fig.add_axes([0.58, 0.6, 0.01, 0.07])

#     cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
#                                 spacing='proportional', boundaries=bounds)

#     ax2.text(1.2, .05, 'NO DATA')
#     ax2.text(1.2, .39, 'NO C. DIFF DETECTED')
#     ax2.text(1.2, .73, 'C. DIFF DETECTED')

#     ax2.tick_params(which='major', length=0)
#     ax2.set_yticklabels(['', '', ''], minor=False)

#     # Major ticks
#     ax.grid(b=None, which='major')
#     ax.set_xticks(np.arange(0, len(np.arange(0, 8.5, .5)), 1), minor=False)
#     ax.set_xticklabels(np.arange(0, 8.5, .5), fontsize=18, minor=False)

#     ax.set_yticks(np.arange(.5, len(cd.pt_info_dict.keys())+.5, 1), minor=True)
#     ax.set_xticks(np.arange(0+.5, len(np.arange(0, 8.5, .5))+.5, 1), minor=True)


#     ax.grid(color='w', linestyle='-', linewidth=2, which='minor')
#     ax.set_yticks(np.arange(img.shape[0]))
#     ax.set_yticklabels(np.arange(img.shape[0]), fontsize=20)
#     ax.set_xlabel('WEEKS', fontsize=25)
#     ax.set_ylabel('PATIENTS', fontsize=25)

#     plt.tight_layout()
#     plt.savefig(path + 'pt_summary.png')
#     # plt.colorbar()

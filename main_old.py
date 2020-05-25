from dataLoaderCdiff import *
import scipy.stats as st
from collections import Counter
from ml_methods import *
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse
import pickle


def main(path, ml, lambdas):

    epochs = 1000

    reg_vec = [None, 1, 2]
    auc_all = []
    auc_all_std = []
    barlabs = []
    cvec = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    weight_vec = [False, True]

    outer_loops = 5
    dattype = 'all_data'
    # for dattype in list(ml.targets_dict.keys()):
    auc_all = []
    auc_all_std = []
    barlabs = []
    np.random.seed(4)
    labels = ml.targets_dict[dattype]
    labels = (np.array(labels) == 'Recur').astype('float')
    # if np.sum(labels)==0 or np.sum(labels)==len(labels):
    #         continue

    fix, ax = plt.subplots(len(reg_vec), len(weight_vec), figsize=(20, 20))
    kk = 0

    results_dict = {}
    for ii, reg in enumerate(reg_vec):
        for jj, ww in enumerate(weight_vec):

            if dattype == 'all_data':
                data_in = ml.new_info_dict
            else:
                data_in = np.array(ml.data_dict[dattype])

            dkey = str(ww) + '_' + str(reg)
            results_dict[dkey] = {}
            auc_vec = []
            if reg is not None:
                fig2, ax2 = plt.subplots(1, outer_loops, figsize=(50, 20))
                fig2.suptitle('Weight ' + str(ww) +
                              ', regularization l' + str(reg), fontsize=40)

            results_dict[dkey]['inner_loss'] = []
            results_dict[dkey]['y_guess'] = []
            results_dict[dkey]['y_true'] = []
            results_dict[dkey]['net'] = []
            results_dict[dkey]['best_lambda'] = []
            for i in range(outer_loops):
                net = LogRegNet(ml.data_dict[dattype].shape[1])
                # net, epochs, labels, data, folds=3, regularizer=None, weighting=True, lambda_grid=None
                inner_loss, y_guess, y_true, net, best_lambda = ml.train_net(
                    net, epochs, labels, data_in, regularizer=reg, weighting=ww, lambda_grid=lambda_vector)
                results_dict[dkey]['inner_loss'].append(inner_loss)
                results_dict[dkey]['y_guess'].append(y_guess)
                results_dict[dkey]['y_true'].append(y_true)
                results_dict[dkey]['net'].append(net)
                results_dict[dkey]['best_lambda'].append(best_lambda)
                if inner_loss is not None:
                    for k in inner_loss.keys():
                        ax2[i].scatter([k]*len(inner_loss[k]),
                                       inner_loss[k], s=150)
                        ax2[i].set_xlabel('lambda values', fontsize=30)
                        ax2[i].set_ylabel('Loss', fontsize=30)
                        ax2[i].set_xscale('log')
                        ax2[i].set_title('Outer Fold ' + str(i), fontsize=30)

                    fig2.savefig('lambdas_w' + str(ww) +
                                 '_l' + str(reg) + '.png')
                    fig2.show()
                fpr, tpr, _ = roc_curve(y_true, y_guess[:, 1].squeeze())
                roc_auc = auc(fpr, tpr)
                if i != 0:
                    ax[ii, jj].plot(fpr, tpr, alpha=0.7, color=cvec[kk])
                else:
                    if reg is not None:
                        reglab = ', l'+str(reg)
                        if ww is True:
                            reglab = ', balanced weights' + ', l'+str(reg)
                        ax[ii, jj].plot(fpr, tpr, color=cvec[kk], alpha=.7, label=reglab[2:] +
                                        ', AUC = ' + str(np.round(np.mean(auc_vec[-25:]), 3)))
                    else:
                        reglab = ''
                        if ww is True:
                            reglab = ', balanced weights'
                        ax[ii, jj].plot(fpr, tpr, color=cvec[kk], alpha=.7, label='no regularization' +
                                        reglab + ', AUC = ' + str(np.round(np.mean(auc_vec[-25:]), 3)))

                auc_vec.append(roc_auc)
                print('loop ' + str(i) + ' complete')

            auc_all.append(np.mean(auc_vec))
            auc_all_std.append(np.std(auc_vec))

            if dattype == 'week_one':
                ax[ii, jj].set_title('ROC Curves, Week 1, Eventual Reurrence' + reglab +
                                    ', AUC = ' + str(np.round(np.mean(auc_vec[-25:]), 3)), fontsize=15)
                barlabs.append('Week 1 eventual recurr' + reglab)
            elif dattype == 'all_data':
                ax[ii, jj].set_title('ROC Curves, ' + (str(dattype).replace('_', ' ')).capitalize(
                ) + reglab + ', AUC = ' + str(np.round(np.mean(auc_vec[-25:]), 3)), fontsize=15)
                barlabs.append('All Data' + reglab)
            else:
                ax[ii, jj].set_title('ROC Curves, Week ' + str(dattype) + reglab +
                                    ', AUC = ' + str(np.round(np.mean(auc_vec[-25:]), 3)), fontsize=15)
                barlabs.append('Week ' + str(dattype)+reglab)
            ax[ii, jj].plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax[ii, jj].set_xlabel('False Positive Rate')
            ax[ii, jj].set_ylabel('True Positive Rate')

            kk += 1
    #             plt.legend()
    plt.savefig('nested_lr_AUC' + str(dattype).replace('.', '_') + '.png')
    fix.show()

    plt.figure()
    plt.bar(np.arange(len(auc_all)), auc_all, yerr=auc_all_std)
    plt.xticks(np.arange(len(auc_all)), barlabs)
    plt.ylim([0, 1])
    plt.xticks(rotation=45, rotation_mode='anchor', horizontalalignment='right')
    plt.title("Average AUC score of 5 Outer CV Loops, Week " +
            str(dattype).replace('.', '_'))
    plt.tight_layout()
    plt.savefig('nested_lr2_avgAUC' +
                str(dattype).replace('.', '_') + '.png')

    ff = open("output.pkl", "wb")
    pickle.dump(results_dict, ff)
    ff.close()

if __name__ == "__main__":
    path = '/PHShome/jjd65/CDIFF/cdiff_metabolomics/outdir/'
    cd = cdiffDataLoader()
    cd.make_pt_dict(cd.cdiff_raw)
    filt_out = cd.filter_metabolites(40)

    lambda_vector = np.logspace(-6,4,num = 100)

    ml = mlMethods(cd.pt_info_dict, lag=1)
    ml.path = path
    main(path, ml, lambda_vector)

from dataLoaderCdiff import *
import scipy.stats as st
from collections import Counter
from ml_methods import *
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse
import pickle
import os


def main(path, ml, lambdas, inner_fold=True, optim_param='auc', dattype='all_data', perc = None, inner_loo = True, outer_loo = True, folds = 5):

    epochs = 1000

    reg_vec = [1, 2, None]
    auc_all = []
    auc_all_std = []
    barlabs = []
    cvec = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    weight_vec = [False, True]

    data_in = ml.data_dict[dattype]
    targets = ml.targets_int[dattype]

    if outer_loo == True:
        ixx = ml.leave_one_out_cv(data_in, targets)
        outer_loops = len(ixx)
    else:
        outer_loops = folds
    
    # for dattype in list(ml.targets_dict.keys()):
    auc_all = []
    auc_all_std = []
    barlabs = []
    np.random.seed(4)
    labels = targets
    # labels = (np.array(labels) == 'Recur').astype('float')
    # if np.sum(labels)==0 or np.sum(labels)==len(labels):
    #         continue
    fix, ax = plt.subplots(len(reg_vec), len(weight_vec), figsize=(20, 20))
    kk = 0

    results_dict = {}
    for ii, reg in enumerate(reg_vec):
        for jj, ww in enumerate(weight_vec):

            if reg is not None:
                reglab = ', l'+str(reg)
                if ww is True:
                    reglab = ', balanced weights' + ', l'+str(reg)
            else:
                reglab = 'no regularization'
                if ww is True:
                    reglab = ', balanced weights, no regularization'
                    

            if isinstance(lambdas, dict):
                lambda_in = lambdas['w'+ str(ww) + '_l' + str(reg)]
            else:
                lambda_in = lambdas
            

            dkey = str(ww) + '_' + str(reg)
            results_dict[dkey] = {}
            auc_vec = []
            results_dict[dkey]['inner_dic'] =[]
            results_dict[dkey]['y_guess'] = []
            results_dict[dkey]['y_true'] = []
            results_dict[dkey]['best_lambda'] = []
            results_dict[dkey]['net'] = []
            results_dict[dkey]['metabs1'] = []
            results_dict[dkey]['metabs2'] = []
            results_dict[dkey]['outer_run'] = []

            if reg is not None and inner_fold is True:
                fig2, ax2 = plt.subplots(figsize = (50,20))
                fig2.suptitle('Weight ' + str(ww) + ', regularization l' + str(reg), fontsize = 40)
                

            for ol in range(outer_loops):
                
                # net, epochs, labels, data, folds=3, regularizer=None, weighting=True, lambda_grid=None
                # inner_auc, y_guess_fin, y_true, net_out, best_lambda

                if outer_loo:
                    ix_in = ixx[ol]
                else:
                    ix_in = None
                inner_dic, y_guess, y_true, net_out, best_lambda, outer_run = ml.train_net(
                    epochs, labels, data_in, regularizer=reg, weighting=ww, lambda_grid=lambda_in, 
                    train_inner = inner_fold, optimization = optim_param, perc = perc, ixs = ix_in, loo_outer = outer_loo, loo_inner = inner_loo)
                weights = [param for param in net_out.parameters()]
            
                metab_ixs = np.argsort(np.abs((weights[0][1,:].T).detach().numpy()))
                metabs =data_in.columns.values[metab_ixs]
                vals = np.sort(np.abs((weights[0][1,:].T).detach().numpy()))
                vals = (vals - vals.min())/(vals.max()-vals.min())
                
                metab_ixs2 = np.argsort(np.abs((weights[0][1,:].T-weights[0][1,:].T).detach().numpy()))
                metabs2 = data_in.columns.values[metab_ixs2]
                # import pdb; pdb.set_trace()
                vals2 = np.sort(np.abs((weights[0][1,:].T-weights[0][0,:].T).detach().numpy()))
                vals2 = (vals2 - vals2.min())/(vals2.max()-vals2.min())

                
                results_dict[dkey]['inner_dic'].append(inner_dic)
                results_dict[dkey]['y_guess'].append(y_guess)
                results_dict[dkey]['y_true'].append(y_true)
                results_dict[dkey]['net'].append(net_out)
                results_dict[dkey]['best_lambda'].append(best_lambda)
                results_dict[dkey]['metabs1'].append(pd.DataFrame(metabs, vals))
                results_dict[dkey]['metabs2'].append(
                    pd.DataFrame(metabs2, vals2))
                results_dict[dkey]['outer_run'].append(outer_run)
                # import pdb; pdb.set_trace()
                # cvec = ['c','m','g']
                if inner_dic is not None and inner_fold is True:
                    for ij,k in enumerate(inner_dic.keys()):
                        ax2.scatter([k],
                                        [inner_dic[k][optim_param]], s=150)
                        ax2.set_xlabel('lambda values', fontsize=30)
                        ax2.set_ylabel(optim_param.capitalize(), fontsize=30)
                        ax2.set_xscale('log')
                        ax2.set_title('Outer Fold ' + str(ol), fontsize=30)
                        if optim_param == 'auc' or optim_param == 'f1':
                            ax2.set_ylim(0,1)

                    # fig2.savefig(optim_param + '_lambdas_w' + str(ww) + '_l' + str(reg) + '.png')
                if not outer_loo:
                    fpr, tpr, _ = roc_curve(y_true, y_guess[:, 1].squeeze())
                    roc_auc = auc(fpr, tpr)
                    auc_vec.append(roc_auc)
                    if ol ==0:
                        ax[ii, jj].plot(fpr, tpr, alpha=0.7, color=cvec[kk])
                    if ol > 0:
                        ax[ii, jj].plot(fpr, tpr, color=cvec[kk], alpha=.7, label=reglab[2:] +
                                        ', AUC = ' + str(np.round(np.mean(auc_vec[-25:]), 3)))
            
                print('loop ' + str(ol) + ' complete')
            if outer_loo:
                y_true = np.concatenate(results_dict[dkey]['y_true'])
                y_guess = np.concatenate(results_dict[dkey]['y_guess'])
                fpr, tpr, _ = roc_curve(y_true, y_guess[:, 1].squeeze())
                roc_auc = auc(fpr, tpr)
                auc_vec.append(roc_auc)
                ax[ii, jj].plot(fpr, tpr, color=cvec[kk], alpha=.7, label=reglab[2:] +
                                ', AUC = ' + str(roc_auc))


            auc_all.append(np.mean(auc_vec))
            auc_all_std.append(np.std(auc_vec))
            if inner_dic is not None and inner_fold is True:
                fig2.savefig(path + dattype + '_' + optim_param + '_lambdas_w' + str(ww) + '_l' + str(reg) + '.png')

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
    plt.savefig(path + dattype + '_' + optim_param + '_nested_lr' + str(dattype).replace('.', '_') + '.png')

    plt.figure(figsize=(18, 10))
    plt.bar(np.arange(len(auc_all)), auc_all, yerr=auc_all_std)
    plt.xticks(np.arange(len(auc_all)), barlabs, fontsize = 20)
    plt.ylim([0, 1])
    plt.yticks(np.linspace(0,1,11), fontsize = 20)
    plt.xticks(rotation=45, rotation_mode='anchor', horizontalalignment='right', fontsize = 20)
    if dattype == 'all_data':
        plt.title("Average AUC score of " + str(folds) + " Outer CV Loops, All Data", fontsize = 25)
    else: 
        plt.title("Average AUC score of 5 Outer CV Loops, Week " + str(dattype), fontsize=25)
    plt.tight_layout()
    plt.savefig(path + dattype + '_' + optim_param + '_nested_lr2_avgAUC' +
                str(dattype).replace('.', '_') + '.png')

    ff = open(path + dattype + '_' + optim_param + "_output.pkl", "wb")
    pickle.dump(results_dict, ff)
    ff.close()

if __name__ == "__main__":
    # path = '/PHShome/jjd65/CDIFF/cdiff_metabolomics/outdir/'
    parser = argparse.ArgumentParser()
    cd = cdiffDataLoader()
    cd.make_pt_dict(cd.cdiff_raw)
    filt_out = cd.filter_metabolites(40)

    lambda_vector = np.logspace(-3, 2, num = 50)

    parser.add_argument("-o", "--optim_type", help="type of lambda optimization", type=str)
    args = parser.parse_args()

    parser.add_argument("-dtype", "--data_type",
                        help="type of lambda optimization", type=str)
    args = parser.parse_args()

    path = 'outputs_june13/'

    ml = mlMethods(cd.pt_info_dict, lag=1)
    ml.path = path

    # main(path, ml, lambda_vector, optim_param = 'loss', dattype = 'week_one')
    main(path, ml, lambda_vector, optim_param = args.optim_type, dattype = args.data_type)


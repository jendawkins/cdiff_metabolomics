from dataLoaderCdiff import *
import scipy.stats as st
from collections import Counter
from ml_methods import *
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse
import pickle
import os
from training_outer import *
import seaborn as sn
import pandas as pd


def main_outer(path, ml, optim_param='auc', dattype='all_data', perc = None, inner_loo = False, outer_loo = True, folds = 5):

    epochs = 100

    optimization = optim_param
    # reg_vec = [1, 2, None]
    reg_vec = [1, None]
    auc_all = []
    auc_all_std = []
    barlabs = []
    cvec = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

###########
    weight_vec = [True, False]
    ww = True
    jj = 0

    data_in = ml.data_dict[dattype]
    targets = ml.targets_int[dattype]

    # temp = np.where(ml.targets_int[dattype] == 1)[0]
    # ixs = temp[2:]
    # ixkeep = list(set(range(data_in.shape[0])) - set(ixs))
    # # print(ixkeep)
    # # print(ixs)
    # data_in = data_in.iloc[np.array(ixkeep),:]
    # targets = targets[np.array(ixkeep)]

    # targets = np.abs(np.array(targets) -1)
    if outer_loo == True:
        ixx = ml.leave_one_out_cv(data_in, targets)
        outer_loops = len(ixx)
    else:
        outer_loops = folds
    
    inner_fold = False
    auc_all = []
    f1_vec = []
    auc_all_std = []
    barlabs = []

    tpr_vec = []
    fpr_vec = []
    np.random.seed(4)
    labels = targets

    tpr_r_vec = []
    fpr_r_vec = []

    kk = 0

    results_dict = {}
    for ii, reg in enumerate(reg_vec):

        for jj, ww in enumerate(weight_vec):

            if reg is not None:
                reglab = 'l'+str(reg)
                if ww is True:
                    reglab = 'balanced' + ', l'+str(reg)
            else:
                reglab = 'no balancing, no regularization'
                if ww is True:
                    reglab = 'balanced, no regularization'
                    
            if reg is not None:
                with open(path + dattype + '_' + str(ww) + '_' + str(reg) + 'inner_dic.pkl', 'rb') as f:
                    inner_dic = pickle.load(f)
            

                if optim_param == 'loss':
                    max_val = np.min([inner_dic[it][optim_param]
                                    for it in inner_dic.keys()])
                else:
                    max_val = np.max([inner_dic[it][optim_param]
                                    for it in inner_dic.keys()])
                best_lambda = [inner_dic[k][optimization]
                            for k in inner_dic.keys() if inner_dic[k][optimization] == max_val]
                # print('Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weighting) + 'and l'+ str(regularizer))
                best_lambda = np.median(best_lambda)
            else:
                best_lambda = None
                inner_dic = None

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
            results_dict[dkey]['pred_lr'] = []

            if reg is not None:
                fig2, ax2 = plt.subplots(figsize = (50,20))
                fig2.suptitle('Weight ' + str(ww) + ', regularization l' + str(reg), fontsize = 40)
                
            # import pdb; pdb.set_trace()
            tprr_vec = []
            fprr_vec = []

            for ol in range(outer_loops):
                
                if outer_loo:
                    ix_in = ixx[ol]
                else:
                    ix_in = None

                y_guess, y_true, net_out, best_lambda, outer_run, pred_lr = train_net(ml, 
                    epochs, labels, data_in, regularizer=reg, weighting=ww, lambda_grid=best_lambda,
                    train_inner = False, perc = perc, ixs = ix_in, loo_outer = outer_loo, loo_inner = inner_loo)
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
                results_dict[dkey]['pred_lr'].append(pred_lr)

                if inner_dic is not None:
                    for ij,k in enumerate(inner_dic.keys()):
                        ax2.scatter([k],
                                        [inner_dic[k][optim_param]], s=600)
                        ax2.set_xlabel('lambda values', fontsize=30)
                        ax2.set_ylabel(optim_param.capitalize(), fontsize=30)
                        ax2.set_xscale('log')
                        ax2.set_title('Outer Fold ' + str(ol), fontsize=30)
                        if optim_param == 'auc' or optim_param == 'f1':
                            ax2.set_ylim(0,1)

                    # fig2.savefig(optim_param + '_lambdas_w' + str(ww) + '_l' + str(reg) + '.png')
                if not outer_loo:
                    fpr, tpr, thresholds = roc_curve(y_true, y_guess[:, 1].squeeze())
                    roc_auc = auc(fpr, tpr)
                    auc_vec.append(roc_auc)
                    y_pred = np.argmax(y_guess, 1)
                    tprr = len(set(np.where(y_pred == 1)[0]).intersection(
                        set(np.where(y_true == 1)[0])))/len(np.where(y_true == 1)[0])
                    fprr = len(set(np.where(y_pred == 0)[0]).intersection(
                        set(np.where(y_true == 0)[0])))/len(np.where(y_true == 0)[0])

                    tprr_vec.append(tprr)
                    fprr_vec.append(fprr)

            
                print('loop ' + str(ol) + ' complete')
            if outer_loo:
                
                y_true = np.concatenate(results_dict[dkey]['y_true'])
                y_guess = np.concatenate(results_dict[dkey]['y_guess'])
                y_pred = np.argmax(y_guess, 1)

                y_pred_lr = np.concatenate(results_dict[dkey]['pred_lr'])
                # import pdb; pdb.set_trace()
                tprr = len(set(np.where(y_pred == 1)[0]).intersection(
                    set(np.where(y_true == 1)[0])))/len(np.where(y_true == 1)[0])
                fprr = len(set(np.where(y_pred == 0)[0]).intersection(
                    set(np.where(y_true == 0)[0])))/len(np.where(y_true == 0)[0])
                pos = len(np.where(y_true == 1)[0])
                neg = len(np.where(y_true == 0)[0])
                tp = len(set(np.where(y_pred == 1)[0]).intersection(
                    set(np.where(y_true == 1)[0])))
                tn = len(set(np.where(y_pred == 0)[0]).intersection(
                    set(np.where(y_true == 0)[0])))
                fn = pos - tp
                fp = neg - tn
                arr = [[tp, fn], [fp, tn]]

                fig3, ax3 = plt.subplots()
                df_cm = pd.DataFrame(arr, index = ['Actual Recur','Actual Cleared'],columns = ['Predicted Recur','Predicted Cleared'])
                sn.set(font_scale=1.4)  # for label size
                chart = sn.heatmap(df_cm, annot=True, annot_kws={"size": 24})  # font size
                ax3.set_yticklabels(['Actual Recur', 'Actual Cleared'],rotation=45)
                ax3.xaxis.tick_top()
                ax3.xaxis.set_label_position('top')
                ax3.tick_params(length=0)
                plt.title((dattype.replace('_', ' ')).capitalize() + reglab)
                plt.show()
                # import pdb; pdb.set_trace()

                tprr_r = len(set(np.where(y_pred_lr == 1)[0]).intersection(
                    set(np.where(y_true == 1)[0])))/len(np.where(y_true == 1)[0])
                fprr_r = len(set(np.where(y_pred_lr == 0)[0]).intersection(
                    set(np.where(y_true == 0)[0])))/len(np.where(y_true == 0)[0])
                pos = len(np.where(y_true == 1)[0])
                neg = len(np.where(y_true == 0)[0])
                tp = len(set(np.where(y_pred_lr == 1)[0]).intersection(
                    set(np.where(y_true == 1)[0])))
                tn = len(set(np.where(y_pred_lr == 0)[0]).intersection(
                    set(np.where(y_true == 0)[0])))
                fn = pos - tp
                fp = neg - tn
                arr = [[tp, fn], [fp, tn]]

                fig3, ax3 = plt.subplots()
                df_cm = pd.DataFrame(arr, index=['Actual Recur', 'Actual Cleared'], columns=[
                                     'Predicted Recur', 'Predicted Cleared'])
                sn.set(font_scale=1.4)  # for label size
                chart = sn.heatmap(df_cm, annot=True, annot_kws={
                                   "size": 24})  # font size
                ax3.set_yticklabels(
                    ['Actual Recur', 'Actual Cleared'], rotation=45)
                ax3.xaxis.tick_top()
                ax3.xaxis.set_label_position('top')
                ax3.tick_params(length=0)
                plt.title('Sklearn ' +(dattype.replace('_', ' ')).capitalize() + reglab)
                plt.show()

                try:
                    fpr, tpr, _ = roc_curve(y_true, y_guess[:, 1].squeeze())
                except:
                    import pdb; pdb.set_trace()
                roc_auc = auc(fpr, tpr)
                auc_vec.append(roc_auc)
    ################

                f1 = sklearn.metrics.f1_score(y_true, np.argmax(y_guess,1))
                f1_vec.append(f1)

            if not outer_loo:
                tprr = np.mean(tprr_vec)
                fprr = np.mean(fprr_vec)
            auc_all.append(np.mean(auc_vec))
            auc_all_std.append(np.std(auc_vec))
            tpr_vec.append(tprr)
            fpr_vec.append(fprr)

            tpr_r_vec.append(tprr_r)
            fpr_r_vec.append(fprr_r)
            
            if inner_dic is not None and inner_fold is True:
                fig2.savefig(dattype + '_' + optim_param + '_lambdas_w' + str(ww) + '_l' + str(reg) + '.png')

            if dattype == 'week_one':
                barlabs.append(reglab)
            elif dattype == 'all_data':
                barlabs.append(reglab)
            else:
                barlabs.append(reglab)
            kk += 1

    plt.figure()
    bac = (np.array(fpr_vec) + np.array(tpr_vec)) / 2

    plt.bar(np.arange(len(tpr_vec)), tpr_vec, alpha = 0.5, label = 'True Pos Rate', width = .25, align = 'edge')
    plt.bar(np.arange(len(fpr_vec)), bac, alpha=0.5, label='BAC', align = 'center', width = .25)

    plt.bar(np.arange(len(fpr_vec)), fpr_vec,
            alpha=0.5, label='True Neg Rate', width=-.25, align = 'edge')
    plt.xticks(np.arange(len(fpr_vec)), barlabs, fontsize = 20)
    plt.ylim([0, 1])
    plt.yticks(np.linspace(0,1,11), fontsize = 20)
    plt.xticks(rotation=45, rotation_mode='anchor',
               horizontalalignment='right', fontsize=20)
    if dattype == 'all_data':
        plt.title("TPR and TNR, All Data", fontsize = 25)
    else:
        plt.title("TPR and TNR, Week " + str(dattype), fontsize=25)

    plt.legend()
    plt.show()
    plt.savefig(dattype + '_' + optim_param + '_TPR-TNR' +
                str(dattype).replace('.', '_') + '.png')
       
       
    plt.figure()
    bac = (np.array(fpr_r_vec) + np.array(tpr_r_vec)) / 2
    # for cc,ba in enumerate(bac):
    #     if cc == 0:
    #         plt.hlines(.75 + cc,1.25 + cc,ba, label = 'BAC')
    #     else:
    #         plt.hlines(.75 + cc, 1.25 + cc, ba)
    plt.bar(np.arange(len(tpr_vec)), tpr_r_vec, alpha=0.5,
            label='True Pos Rate', width=.25, align='edge')
    plt.bar(np.arange(len(fpr_vec)), bac, alpha=0.5,
            label='BAC', align='center', width=.25)

    plt.bar(np.arange(len(fpr_vec)), fpr_r_vec,
            alpha=0.5, label='True Neg Rate', width=-.25, align='edge')
    plt.xticks(np.arange(len(fpr_r_vec)), barlabs, fontsize=20)
    plt.ylim([0, 1])
    plt.yticks(np.linspace(0, 1, 11), fontsize=20)
    plt.xticks(rotation=45, rotation_mode='anchor',
               horizontalalignment='right', fontsize=20)
    if dattype == 'all_data':
        plt.title("Sklearn TPR and TNR, All Data", fontsize=25)
    else:
        plt.title("Sklearn TPR and TNR, Week " + str(dattype), fontsize=25)
    # plt.tight_layout()
    plt.legend()
    plt.show()
    # import pdb; pdb.set_trace()


    # plt.figure(figsize=(18, 10))
    # plt.bar(np.arange(len(auc_all)), auc_all, yerr=auc_all_std)
    # plt.xticks(np.arange(len(auc_all)), barlabs, fontsize = 20)
    # plt.ylim([0, 1])
    # plt.yticks(np.linspace(0,1,11), fontsize = 20)
    # plt.xticks(rotation=45, rotation_mode='anchor', horizontalalignment='right', fontsize = 20)
    # if dattype == 'all_data':
    #     plt.title("Average AUC score of " + str(folds) + " Outer CV Loops, All Data", fontsize = 25)
    # else: 
    #     plt.title("Average AUC score of 5 Outer CV Loops, Week " + str(dattype), fontsize=25)
    # plt.tight_layout()
    # plt.savefig(dattype + '_' + optim_param + '_nested_lr2_avgAUC' +
    #             str(dattype).replace('.', '_') + '.png')

    # plt.figure(figsize=(18, 10))
    # plt.bar(np.arange(len(auc_all)), f1_vec)
    # plt.xticks(np.arange(len(auc_all)), barlabs, fontsize=20)
    # plt.ylim([0, 1])
    # plt.yticks(np.linspace(0, 1, 11), fontsize=20)
    # plt.xticks(rotation=45, rotation_mode='anchor',
    #            horizontalalignment='right', fontsize=20)
    # if dattype == 'all_data':
    #     plt.title("F1 score of " + str(folds) +
    #               " Outer CV Loops, All Data", fontsize=25)
    # else:
    #     plt.title("F1 score of 5 Outer CV Loops, Week " +
    #               str(dattype), fontsize=25)
    # plt.tight_layout()
    # plt.savefig(dattype + '_' + optim_param + '_nested_F1' +
    #             str(dattype).replace('.', '_') + '.png')

    ff = open(path + dattype + '_' + optim_param + "_output.pkl", "wb")
    pickle.dump(results_dict, ff)
    ff.close()

if __name__ == "__main__":
    # path = '/PHShome/jjd65/CDIFF/cdiff_metabolomics/outdir/'
    parser = argparse.ArgumentParser()
    cd = cdiffDataLoader()
    # cd.make_pt_dict(cd.cdiff_raw)
    # filt_out = cd.filter_metabolites(40)
    path = 'outputs_june18/'

    # lambda_vector = np.logspace(-3, 2, num=3)

    parser.add_argument("-o", "--optim_type", help="type of lambda optimization", type=str)

    parser.add_argument("-dtype", "--data_type",
                        help="type of lambda optimization", type=str)

    
    args = parser.parse_args()

    import os
    if not os.path.isdir(path):
        os.mkdir(path)

    ml = mlMethods(cd.pt_info_dict, lag=1)
    ml.path = path

    # main(path, ml, lambda_vector, optim_param = 'loss', dattype = 'week_one')
    main_outer(path, ml, optim_param = args.optim_type, dattype = args.data_type)


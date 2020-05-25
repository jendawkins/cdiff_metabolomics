import scipy.stats as st
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch.nn as nn
from torch.nn import functional as F
import torch
from sklearn.feature_selection import SelectFromModel
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as T
from sklearn.metrics import roc_curve, auc, roc_auc_score

class Net(nn.Module):
    def __init__(self, num_mols, hidden_size):
        super(Net,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_mols, hidden_size),
            nn.BatchNorm1d(hidden_size), #applying batch norm
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_size,2),
        )

    def forward(self,x):
        x = self.calssifier(x).squeeze()
        return x


class LogRegNet(nn.Module):
    def __init__(self, num_mols):
        super(LogRegNet, self).__init__()
        self.linear = nn.Linear(num_mols, 2)

    def forward(self, x):
        x = self.linear(x).squeeze()
        return x


class mlMethods():
    def __init__(self, pt_info_dict_orig, lag = 1, option = 2):
        new_info_dict = copy.deepcopy(pt_info_dict_orig)
        new_info_dict2 = copy.deepcopy(pt_info_dict_orig)
        for patient in new_info_dict:
            if new_info_dict[patient][1.0]['PATIENT STATUS (BWH)']=='Recur':
                tmpts = list(new_info_dict[patient].keys())
                num_labels = len(tmpts)-lag
                labs = ['Cleared']*(num_labels-lag)
                labs.extend(['Recur'])
                for i,lab in enumerate(labs):
                    new_info_dict[patient][tmpts[i]]['PATIENT STATUS (BWH)'] = lab
                if lag > 1:
                    for i in range(lag):
                        new_info_dict[patient].pop(tmpts[-i])
                        new_info_dict2[patient].pop(tmpts[-i])
                else:
                    new_info_dict[patient].pop(tmpts[-lag])
                    new_info_dict2[patient].pop(tmpts[-lag])
            else:
                tmpts = list(new_info_dict[patient].keys())
                num_labels = len(tmpts)
                labs = ['Cleared']*(num_labels)
                for i,lab in enumerate(labs):
                    new_info_dict[patient][tmpts[i]]['PATIENT STATUS (BWH)'] = lab
            pt_info_dict = new_info_dict
            pt_info_dict2 = new_info_dict2
        

        tmpts = [list(pt_info_dict[i].keys())
                for i in pt_info_dict.keys()]
        all_pts = np.unique([inner for outer in tmpts for inner in outer])
        self.tmpts = all_pts
        self.week = dict()
        self.targets_dict = dict()
        self.targets_dict2 = dict()
        for pts in all_pts:
            self.week[pts] = pd.concat([pt_info_dict[i][pts]['DATA']
                                for i in pt_info_dict.keys()
                                if pts in pt_info_dict[i].keys()], 1).T
            self.targets_dict[pts] = [pt_info_dict[i][pts]['PATIENT STATUS (BWH)'] 
                                for i in pt_info_dict.keys()
                                if pts in pt_info_dict[i].keys()]

            self.targets_dict2[pts] = [pt_info_dict2[i][pts]['PATIENT STATUS (BWH)']
                                           for i in pt_info_dict2.keys()
                                           if pts in pt_info_dict2[i].keys()]

        days = [sorted(list(pt_info_dict[i].keys()))
                for i in pt_info_dict.keys()]

        N = 0.0
        days = [[day for day in sub if day != N] for sub in days]
        all_data = []
        labels = []
        
        for i in pt_info_dict.keys():
            
            if pt_info_dict[i] and len(days[i]) > 0:
                labels.extend([pt_info_dict[i][days[i][k]]['PATIENT STATUS (BWH)'] for
                            k in range(len(days[i]))])
                all_data.append(pd.concat(
                    [pt_info_dict[i][days[i][k]]['DATA'] for k in range(len(days[i]))], 1))

        all_data = pd.concat(all_data, 1).T
        self.targets_dict['all_data'] = labels
        self.targets_dict['week_one'] = self.targets_dict2[1.0]

        self.week['all_data'] = all_data
        self.week['week_one'] = self.week[1.0]
        # self.week['all'] = pd.concat(all_data, 1).T
        # self.targets_orig = [pt_info_dict[i][1.0]
        #                 ['PATIENT STATUS (BWH)'] for i in pt_info_dict.keys()]
        # self.targets_all_orig = labels

        lst = list(self.week.keys())
        self.data_dict_raw = {}
        self.data_dict = {}
        self.data_dict_raw_filt = {}
        self.targets_int = {}
        self.data_dict_log = {}
        for ls in lst:
            logdat = self.log_transform(self.week[ls])
            self.data_dict_log[ls] = logdat
            filtdata = self.filter_vars(logdat, self.targets_dict[ls], var_type = 'between')
            rawfilt = self.filter_vars(
                self.week[ls], self.targets_dict[ls], var_type='between')
            self.data_dict_raw_filt[ls] = rawfilt
            self.data_dict[ls] = filtdata
            self.data_dict_raw[ls] = self.week[ls]
            self.targets_int[ls] = (
                np.array(self.targets_dict[ls]) == 'Recur').astype('float')

        self.new_info_dict = new_info_dict

        self.path = 'figs/'


    def make_pt_dict(self, data, idxs=None):
        header_lst = [i[4:]
                      for i in data.columns.values if 'Unnamed' not in i[0]]
        header_names = header_lst
        header_labels = ['CLIENT SAMPLE ID','CLIENT MATRIX','SAMPLE DESCRIPTION',
        'CULTURE STATUS (BWH)','PATIENT STATUS (BWH)','MASS EXTRACTED']

        # pt_info_dict = {header_names[j][0]:{header_labels[i]:header_names[j][i] for i in range(1,len(header_labels))} for j in range(len(header_names))}

        row_names = data.index.values

        pt_names = np.array([h[0].split('-')[0] for h in header_names])
        pts = []
        for i, n in enumerate(np.unique(pt_names)):
            pts.extend([str(i+1) + '.' + str(j)
                        for j in range(len(np.where(pt_names == n)[0]))])

        ts = np.array([h[0].split('-')[1] for h in header_names])
        times = []
        for el in ts:
            try:
                times.append(float(el))
            except:
                tm = str(ord(list(el)[1])-97)
                times.append(float(list(el)[0] + '.' + tm))
        idx_val = row_names
        if idxs is None:
            data = pd.DataFrame(
                np.array(data), columns=header_names, index=row_names)
        else:
            data = pd.DataFrame(
                np.array(data)[idxs, :], columns=header_names, index=np.array(row_names)[idxs])

        pt_info_dict = {}
        pt_key = 0
        pt_info_dict[pt_key] = {}

        for j in range(len(header_names)):
            if j != 0 and pt_names[j] != pt_names[j-1]:
                pt_key += 1
                pt_info_dict[pt_key] = {}
                try:
                    pt_info_dict[pt_key][times[j]] = {
                        header_labels[i]: header_names[j][i] for i in range(len(header_labels))}
                except:
                    import pdb; pdb.set_trace()
                pt_info_dict[pt_key][times[j]].update(
                    {'DATA': data.iloc[:, j]})

            else:
                try:
                    pt_info_dict[pt_key][times[j]] = {
                        header_labels[i]: header_names[j][i] for i in range(len(header_labels))}
                except:
                    import pdb; pdb.set_trace()
                pt_info_dict[pt_key][times[j]].update(
                    {'DATA': data.iloc[:, j]})
        return pt_info_dict

    def log_transform(self, data):
        temp = data.copy()
        temp = temp.replace(0,np.inf)
        return np.log(data + 0.5*np.min(temp,0))

    def normalize(self,x):
        assert(x.shape[0]<x.shape[1])
        return (x - np.mean(x, 0))/np.std(x, 0)

    def make_metabolome_info_dict(self, metabolites, metabolome_info_dict):
        metabolome_info_dict_2 = {m: metabolome_info_dict[m] for m in metabolites}
        return metabolome_info_dict_2

    def vars(self,data, labels, normalize_data = False):
        if normalize_data:
            data = self.normalize(data)
        # labels = self.targets_dict[dat_type]
        cleared = data[np.array(labels) == 'Cleared']
        recur = data[np.array(labels) == 'Recur']
        within_class_vars = [np.var(cleared, 0), np.var(recur, 0)]
        class_means = [np.mean(cleared, 0), np.mean(cleared, 0)]

        total_mean = np.mean(data, 0)
        between_class_vars = 0
        for i in range(2):
            between_class_vars += (class_means[i] - total_mean)**2

        total_vars = np.var(data, 0)
        vardict = {'within':within_class_vars,'between':between_class_vars,'total':total_vars}
        return vardict

    def filter_vars(self, data, labels, perc=5, var_type = 'between', normalize_data = False):
        vardict = self.vars(data, labels, normalize_data)
        variances = vardict[var_type]
        variances = variances.replace(np.nan,0)
        rm2 = set(np.where(variances > np.percentile(variances, perc))[0])
        return data.iloc[:,list(rm2)]

    def summarize(self,metabolome_info_dict,pt_info_dict, title, print_summary = True):
        cdict = Counter([metabolome_info_dict[i]['SUPER_PATHWAY']
                         for i in metabolome_info_dict.keys()])

        D = cdict
        labels = list(D.keys())
        labels.sort()
        values = [D[label] for label in labels]
        plt.rcParams['figure.figsize'] = 7,7
        plt.bar(range(len(D)), values, align='center')
        plt.xticks(range(len(D)), list(labels))
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.ylabel('Number of Molecules', fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(title,fontsize = 20)
        plt.tight_layout()
        plt.savefig(self.path + title.replace(' ','_') + 'Counts.png')
        plt.show()

        if print_summary:
            pt_status = [[pt_info_dict[pts][i]['PATIENT STATUS (BWH)'] for i in \
                pt_info_dict[pts].keys(
            )][0] for pts in pt_info_dict.keys()]

            well = sum([p == 'Cleared' for p in pt_status])
            sick = sum([p == 'Recur' for p in pt_status])
            print('Pts with CDIFF: ' + str(sick))
            print('Pts cleared: ' + str(well))

    def split_test_train(self, data, labels, perc=.8, all_data=False):
        if all_data:
            assert(isinstance(data, dict))
            status = [data[i][list(data[i].keys())[-1]]['PATIENT STATUS (BWH)']
                    for i in list(data.keys())]
            classes = np.unique(status)
            c1 = np.where(np.array(status) == classes[0])[0]
            c2 = np.where(np.array(status) == classes[1])[0]
            ixtrain = np.concatenate((np.random.choice(c1, np.int(len(c1)*perc), replace=False),
                                    np.random.choice(c2, np.int(len(c2)*perc), replace=False)))
            ixtest = np.array(list(set(range(len(status))) - set(ixtrain)))
            train = []
            test = []
            train_l = []
            test_l = []
            for i in ixtrain:
                days = [sorted(list(data[ii].keys()))
                        for ii in data.keys()]

                N = 0.0
                days = [[day for day in sub if day != N] for sub in days]
                if data[i] and len(days[i]) > 0:
                    train_l.extend([data[i][days[i][k]]['PATIENT STATUS (BWH)'] for
                                    k in range(len(days[i]))])
                    train.append(pd.concat(
                        [data[i][days[i][k]]['DATA'] for k in range(len(days[i]))], 1))
            for i in ixtest:
                days = [sorted(list(data[ii].keys()))
                        for ii in data.keys()]
                if data[i]:
                    test_l.extend([data[i][days[i][k]]['PATIENT STATUS (BWH)'] for
                                k in range(len(days[i]))])
                    test.append(pd.concat(
                        [data[i][days[i][k]]['DATA'] for k in range(len(days[i]))], 1))
            train = pd.concat(train, 1).T
            test = pd.concat(test, 1).T

            train = self.log_transform(train)
            test = self.log_transform(test)

            train = self.filter_vars(train, train_l, var_type='between')
            test = self.filter_vars(test, test_l, var_type='between')

            train, train_l, test, test_l = np.array(train), np.array(
                train_l), np.array(test), np.array(test_l)
            train_l = np.array(train_l == 'Recur').astype('float')
            test_l = np.array(test_l == 'Recur').astype('float')
        else:
            classes = np.unique(labels)
            c1 = np.where(labels == classes[0])[0]
            c2 = np.where(labels == classes[1])[0]
            ixtrain = np.concatenate((np.random.choice(c1, np.int(len(c1)*perc), replace=False),
                                    np.random.choice(c2, np.int(len(c2)*perc), replace=False)))
            ixtest = np.array(list(set(range(len(labels))) - set(ixtrain)))
            train, train_l, test, test_l = data[ixtrain,
                                                :], labels[ixtrain], data[ixtest, :], labels[ixtest]

        return train, train_l, test, test_l
    
    def LOOCV_split(self,data,labels):
        ix_out = int(np.random.choice(np.arange(len(labels)), 1))

        ix_train = set(np.arange(len(labels)))
        ix_train.remove(ix_out)
        ix_train = np.array(list(ix_train))
        if isinstance(data,pd.DataFrame):
            return data.iloc[ix_train, :], labels[ix_train], data.iloc[ix_out], labels[ix_out], ix_train, ix_out
        else:
            return data[ix_train, :], labels[ix_train], data[ix_out], labels[ix_out], ix_train, ix_out

    def pca_func(self,x,targets,n=2):
        x = (x - np.min(x, 0))/(np.max(x, 0) - np.min(x, 0))
        pca = PCA(n_components=n)
        targets = (np.array(targets) == 'Recur').astype('float')
        if x.shape[0] <= 55:
            title_name = 'Week 1'
        else:
            title_name = 'All'
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
                                'principal component 1', 'principal component 2'])

        variance = pca.explained_variance_ratio_  # calculate variance ratios

        finalDf = pd.concat([principalDf, pd.DataFrame(
            data=np.array(targets), columns=['target'])], axis=1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1, variance explaned= ' +
                    str(np.round(variance[0], 3)), fontsize=15)
        ax.set_ylabel('Principal Component 2, variance explaned= ' +
                    str(np.round(variance[1], 3)), fontsize=15)
        ax.set_title('2 component PCA, ' + title_name + ' Data', fontsize=20)
        targets = np.unique(targets)
        colors = ['r', 'g', 'b']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
        ax.legend(targets)
        ax.grid()
        # return fig, ax
        fig.savefig(self.path + title_name.replace(' ','') + 'pca.pdf')


    def rank_sum(self,x,targets,cutoff = .05):
        targets = (np.array(targets) == 'Recur').astype('float')
        pval = []
        teststat = []
        for i in range(x.shape[1]):
            xin = np.array(x)[:, i]
            X = xin[targets==1]
            Y = xin[targets==0]
            # xin1 = (xin - np.min(xin,0))/(np.max(xin,0)-np.min(xin,0))
            s, p = st.ranksums(X, Y)
            pval.append(p)
            teststat.append(s)
        df = pd.DataFrame(np.vstack((pval, teststat)).T, columns=[
                        'P_Val', 'Test_Stat'], index=x.columns.values)
                
        mols = df.index.values[np.where(np.array(df['P_Val']) < cutoff)[0]]
        # # bonferonni
        bf_cutoff = cutoff / df.shape[0]
        bf_mols = df.index.values[np.where(np.array(df['P_Val']) < bf_cutoff)[0]]

        # # ben-hoch 
        bh_df = df.copy()
        bh_df = bh_df.sort_values('P_Val', ascending=True)
        alphas = (np.arange(1, bh_df.shape[0]+1) / bh_df.shape[0])*cutoff
        out = np.where(bh_df['P_Val'] <= alphas)[0]
        if len(out > 0):
            bh_idx = out[0]
            bh_mols = bh_df['P_Val'].index.values[:bh_idx]
        else:
            bh_mols = []

        # for pv in bh_df['P_Val']:


        return df.sort_values(ascending= True, by = 'P_Val'), mols, bf_mols, bh_mols        
                    


    def log_reg(self,x,targets,features=None,weight = None, regularizer = 'l1', solve = 'liblinear',perc = 10, shuffle = True,maxiter=100, LOOCV = False, folds = 3):
        targets = (np.array(targets)=='Recur').astype('float')
        x = (x - np.min(x,0))/(np.max(x,0)-np.min(x,0))
        if LOOCV:
            X,y,testX,testy,temp1,temp2 = self.LOOCV_split(x, targets)
        else:
            X,y,testX_out,testy_out = self.split_test_train(x,targets)
        if features is not None:
            X = np.array(X[features])
        else:
            X = np.array(X)

        if LOOCV:
            txs = []
            for f in range(folds):
                temp1,temp2,temp3,temp4,train_ixs, test_ix = self.LOOCV_split(X,y)
                txs.append((train_ixs,test_ix))

        skf = StratifiedKFold(n_splits=folds, shuffle = shuffle)

        tst_score = []
        m_all = []
        models = []
        i = 0

        for train_index, test_index in skf.split(X, y):
            if LOOCV:
                train_index, test_index = txs[i]

            X_train, X_test_in = X[train_index,:], X[test_index,:]
            y_train, y_test_in = y[train_index], y[test_index]


            clf = LogisticRegression(class_weight=weight, penalty=regularizer, \
                solver=solve,max_iter=maxiter).fit(X_train , y_train)

            coefs = np.std(X_train, 0)*clf.coef_
            ranks = np.argsort(-np.abs(coefs)).squeeze()
            sorted_coefs = coefs.squeeze()[ranks]
            cutoff = np.where(np.abs(sorted_coefs) <= np.percentile(np.abs(coefs.squeeze()), 100-perc))[0][0]
            mols = self.data_dict[1.0].columns.values[ranks].squeeze()
            if LOOCV:
                X_test = X_test.reshape(1,-1)
                y_test = y_test.reshape(1,-1)
            ts_preds = clf.predict(X_test_in)
            tst_score.append(sklearn.metrics.f1_score(
                y_test_in, ts_preds))
            m_all.append(mols[:cutoff])
            models.append(clf)
            i += 1
            

        best_ix = np.argmax(tst_score)
        mols2 = m_all[best_ix]
        # mols2 = np.unique(np.concatenate(np.array(m_all)[np.array(tst_score) > thresh]))
        # plot_vals_least = mols[:10]
        best_mod = models[best_ix]
        selection_model = SelectFromModel(best_mod, prefit=True)

        xnew = selection_model.transform(X)
        ixs = [int(np.where(np.all((X == np.expand_dims(xnew[:, kk],1)), axis=0))[0]) for kk in range(xnew.shape[1])]
        lasso_mols = x.columns.values[ixs]
        # if LOOCV:
        #     testX = np.array(testX).reshape(1,-1)
        #     testy = testy.reshape(1,-1)
        test_predictions = best_mod.predict(testX_out)
        test_score = best_mod.score(testX_out,testy_out)

        logreg_mols = set(mols2)
        avg_score = np.mean(tst_score)
        pred_probs = best_mod.predict_proba(testX_out)
        return logreg_mols, lasso_mols, avg_score, test_score, test_predictions,testy_out, best_mod, pred_probs

    def ANOVA_F(self,X=None,targets=None,n=10,features=None):
        if X is None:
            X = self.week_one_norm
        if targets is None:
            targets = self.targets
        if features is not None:
            X = X[features]

        y = self.targets
        bestfeatures = SelectKBest(k=n)
        fit = bestfeatures.fit(X, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        #concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
        # print(featureScores.nlargest(n, 'Score'))  # print 10 best features
        sklearn_mols = set(featureScores.nlargest(n, 'Score')['Specs'])
        return sklearn_mols

    def decision_trees(self, X, targets, class_weights = None, n=10):

        targets = (np.array(targets) == 'Recur').astype('float')
        xtrain, ytrain, xtest, ytest = self.split_test_train(X, targets)

        skf = StratifiedKFold(n_splits=3, shuffle = True)

        tst_score = []
        models = []
        r_all = []
        for train_index, test_index in skf.split(xtrain, ytrain):
            X_train, X_test = np.array(xtrain)[train_index,:], np.array(xtrain)[test_index,:]
            y_train, y_test = np.array(ytrain)[train_index], np.array(ytrain)[test_index]

            clf = RandomForestClassifier(class_weight = class_weights).fit(X_train, y_train)

            ixs = np.where(clf.feature_importances_ > 0)[0]
            cutoff = len(ixs)
            ranks = np.argsort(-np.abs(clf.feature_importances_))
            mols_ixs = ranks[:cutoff]
            # mols = self.week_one.columns.values[ranks].squeeze()
            tst_score.append(clf.score(X_test, y_test))

            r_all.append(mols_ixs)
            models.append(clf)

        best_ix = np.argmax(tst_score)
        best_rank = r_all[best_ix]
        mols = X.columns.values[best_rank].squeeze()
        # mols2 = np.unique(np.concatenate(np.array(m_all)[np.array(tst_score) > thresh]))
        # plot_vals_least = mols[:10]
        best_mod = models[best_ix]
        test_preds = best_mod.predict(xtest)
        test_score = best_mod.score(xtest,ytest)

        avg_score = np.mean(tst_score)
        pred_probs = best_mod.predict_proba(xtest)

        return mols, avg_score, test_score, test_preds,ytest, pred_probs

    def corr_fig(self,X,feats,names):
        sns.set(font_scale=3)
        label_encoder = LabelEncoder()
        if len(names)>1:
            for i, poss_good in enumerate(feats):
                poss_good = list(poss_good)
                data = pd.DataFrame(
                    np.hstack((np.expand_dims(self.targets, 1), X[poss_good])))
                poss_good.extend(['T'])
                data.columns = poss_good
                data.iloc[:, 0] = label_encoder.fit_transform(
                    data.iloc[:, 0]).astype('float64')
                corrmat = data.corr()
                # top_corr_features = corrmat.index
                plt.figure(figsize=(30, 30))
                #plot heat map
                g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn", center=0)
                fig = g.get_figure()
                plt.title(names[i])

                fig.savefig(self.path + names[i].replace(' ', '_') + "_corr.png")
                plt.show()
        else:
                poss_good = list(feats)
                data = pd.DataFrame(
                    np.hstack((np.expand_dims(self.targets, 1), X[poss_good])))
                poss_good.extend(['T'])
                data.columns = poss_good
                data.iloc[:, 0] = label_encoder.fit_transform(
                    data.iloc[:, 0]).astype('float64')
                corrmat = data.corr()
                # top_corr_features = corrmat.index
                plt.figure(figsize=(30, 30))
                #plot heat map
                g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn", center=0)
                fig = g.get_figure()
                plt.title(names[i])

                fig.savefig(self.path + names[i].replace(' ', '_') + "_corr.png")
                plt.show()

    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def make_one_hot(self, a, num_classes):
        a = a.numpy()
        b = torch.zeros((a.shape[0], num_classes))
        b[np.arange(a.shape[0]), a] = 1
        return torch.Tensor(b)

    def train_net(self, net, epochs, labels, data, plot = True, folds = 3, shuffle = False, data_type = 'week_one', regularizer = None, weighting = True, lambda_grid=None, train_inner = True, optimization = 'auc'):
        # OUTER TRAIN TEST SPLIT
        TRAIN, TRAIN_L, TEST, TEST_L = self.split_test_train(data,labels, all_data = isinstance(data, dict))
        TRAIN, TRAIN_L, TEST, TEST_L = torch.FloatTensor(TRAIN), torch.DoubleTensor(
            TRAIN_L), torch.FloatTensor(TEST), torch.DoubleTensor(TEST_L)
        optimizer = torch.optim.RMSprop(net.parameters(),lr = .0001)
        
        if regularizer is not None and train_inner:
            skf = StratifiedKFold(n_splits=folds, shuffle= shuffle)
            inner_dic = dict()
            # INNER TRAIN TEST SPLIT
            for train_index, test_index in skf.split(TRAIN, TRAIN_L):
                X_train, X_test = TRAIN[train_index,:], TRAIN[test_index,:]
                y_train, y_test = TRAIN_L[train_index], TRAIN_L[test_index]
                test_running_loss = []
                running_auc = []
                running_f1 = []

                y_guess = []
                y_true = []
                net.apply(self.init_weights)

                if weighting:
                    weights = len(y_train) / (2 * np.bincount(y_train))
                    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights))

                else:
                    criterion = nn.BCEWithLogitsLoss()
                
                for lamb in lambda_grid:
                    
                    # INNER FOLD!!!
                    for epoch in range(epochs):
                        net.train()
                        optimizer.zero_grad()
                        out = net(X_train).double()

                        reg_lambda = lamb
    #                     if regularizer is not None:
                        l2_reg = None
                        for W in net.parameters():
                            if l2_reg is None:
                                l2_reg = W.norm(regularizer)
                            else:
                                l2_reg = l2_reg + W.norm(regularizer)

                        loss = criterion(out, self.make_one_hot(y_train,2)) + reg_lambda * l2_reg

                        loss.backward()
                        optimizer.step()
                        
                        
                        net.eval()
                        test_out = net(X_test).double()
                        
                        test_loss = criterion(test_out, self.make_one_hot(y_test,2))

                        test_out_sig = torch.sigmoid(test_out)
                        
                        y_guess = test_out_sig.detach().numpy()
                        fpr, tpr, _ = roc_curve(y_test, y_guess[:,1].squeeze())
                        roc_auc = auc(fpr, tpr)

                        y_pred = np.argmax(y_guess, 1)

                        f1 = sklearn.metrics.f1_score(y_test, y_pred)
                        
                        test_running_loss.append(test_loss.item())
                        running_auc.append(roc_auc)
                        running_f1.append(f1)

                        if optimization == 'auc':
                            running_vec = running_auc
                        elif optimization == 'f1':
                            running_vec = running_f1
                        elif optimization == 'loss':
                            running_vec = test_running_loss

                        bool_test = np.array([r1 <= r2 for r1, r2 in zip(running_vec[-10:],running_vec[-11:-1])]).all()
                        
                        if epoch > 50 and bool_test:
                            break
                    if lamb not in inner_dic.keys():
                        inner_dic[lamb] = [running_vec[-11]]
                    else:
                        inner_dic[lamb].append(running_vec[-11])
                    
            max_val = np.max([np.median(it) for it in inner_dic.values()])
            best_lambda = [k for k,v in inner_dic.items() if max_val in set(v)][0]
            # print('Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weighting) + 'and l'+ str(regularizer))
        else:
            if regularizer is None:
                best_lambda = 0
            elif regularizer is not None and not train_inner:
                assert(isinstance(lambda_grid, float))
                best_lambda = lambda_grid
            inner_dic = None
            
        if weighting:
    #             amt = np.array([len(np.where(y_train == l)[0]) for l in y_train])
            weights = len(TRAIN_L) / (2 * np.bincount(TRAIN_L))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights))

        else:
            criterion = nn.BCEWithLogitsLoss()
                
        # TRAIN OUTER FOLD
        y_guess = []
        test_running_loss = []
        running_auc = []
        running_f1 = []
        net_vec = []
        net.apply(self.init_weights)
        for epoch in range(epochs):
            net.train()
            optimizer.zero_grad()
            out = net(TRAIN).double()

            reg_lambda = best_lambda
            if regularizer is not None:
                l2_reg = None
                for W in net.parameters():
                    if l2_reg is None:
                        l2_reg = W.norm(regularizer)
                    else:
                        l2_reg = l2_reg + W.norm(regularizer)
            else:
                l2_reg = 0
            loss = criterion(out, self.make_one_hot(TRAIN_L,2)) + reg_lambda * l2_reg

            loss.backward()
            optimizer.step()

            net.eval()
            test_out = net(TEST).double()
            test_loss = criterion(test_out, self.make_one_hot(TEST_L,2))

            test_out_sig = torch.sigmoid(test_out)
            y_guess.append(test_out_sig.detach().numpy())

            test_running_loss.append(test_loss.item())
            net_vec.append(net)

            # calc auc
            fpr, tpr, _ = roc_curve(TEST_L, y_guess[-1][:,1].squeeze())
            roc_auc = auc(fpr, tpr)
            # calck f1 score
            y_pred = np.argmax(y_guess[-1],1)
            
            f1 = sklearn.metrics.f1_score(TEST_L, y_pred)

            running_auc.append(roc_auc)
            running_f1.append(f1)
            
            if optimization == 'auc':
                running_vec = running_auc
            elif optimization =='f1':
                running_vec = running_f1
            elif optimization == 'loss':
                running_vec = test_running_loss

            bool_test = np.array([r1 <= r2 for r1, r2 in zip(running_vec[-10:],running_vec[-11:-1])]).all()
            
            if epoch > 50 and bool_test:
                break
        net_out = net_vec[-11]
        y_guess_fin = y_guess[-11]
        y_true = TEST_L

        return inner_dic, y_guess_fin, y_true, net_out, best_lambda, running_vec

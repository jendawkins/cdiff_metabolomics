from dataLoaderCdiff import *
from ml_methods import *
import argparse
import numpy as np
import torch
from torch import nn
import pickle

def main_parallelized(ml, lamb, zip_ixs, net, optimizer, TRAIN, TRAIN_L, epochs, weighting, regularizer, inner_dic):

    y_test_vec = []
    y_guess_vec = []
    test_running_loss = 0
    for train_index, test_index in zip_ixs:
        # initialize net for each new dataset
        net.apply(ml.init_weights)

        X_train, X_test = TRAIN.iloc[train_index,:], TRAIN.iloc[test_index,:]
        y_train, y_test = TRAIN_L[train_index], TRAIN_L[test_index]
        if isinstance(test_index, int):
            X_train, y_train, X_test, y_test = torch.FloatTensor(np.array(X_train)), torch.DoubleTensor(
                y_train), torch.FloatTensor([np.array(X_test)]), torch.DoubleTensor([[y_test]])
        else:
            X_train, y_train, X_test, y_test = torch.FloatTensor(np.array(X_train)), torch.DoubleTensor(
                y_train), torch.FloatTensor(np.array(X_test)), torch.DoubleTensor(y_test)
        
        if weighting:
            weights = len(y_train) / (2 * np.bincount(y_train))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights))

        else:
            criterion = nn.BCEWithLogitsLoss()

        y_test_per_epoch = []
        y_guess_per_epoch = []

        running_vec = []
        for epoch in range(epochs):

            out = ml.train_loop(X_train, y_train, net,
                            optimizer, criterion, lamb, regularizer)
    
            
            # evaluate on test set
            if epoch % 10 ==0:
                test_out, test_loss, y_guess = ml.test_loop(
                    net, X_test, y_test, criterion)
                
                y_test_per_epoch.append(y_test)
                y_guess_per_epoch.append(y_guess)

                running_vec.append(test_loss)
                if len(running_vec) > 12:
                    bool_test = np.array([r1 >= r2 for r1, r2 in zip(
                            running_vec[-10:], running_vec[-11:-1])]).all()
                # perform early stopping if greater than 50 epochs and if either loss is increasing over the past 10 iterations or auc / f1 is decreasing over the past 10 iterations
                if (len(running_vec) > 12 and bool_test):
                    y_test_vec.append(y_test_per_epoch[-11])
                    y_guess_vec.append(y_guess_per_epoch[-11])
                    test_running_loss += test_loss
                    # add record of lambda and lowest loss or highest auc/f1 associated with that lambda at this epoch
                    break

    if len(y_test_vec) ==1:
        y_test_vec.append(y_test_per_epoch[-11])
        y_guess_vec.append(y_guess_per_epoch[-11])
        test_running_loss += test_loss
    y_guess_mat = np.concatenate(y_guess_vec)
    y_pred_mat = np.argmax(y_guess_mat, 1)
    if len(y_test_vec) < y_guess_mat.shape[0]:
        y_test_vec = np.concatenate(y_test_vec)
    f1 = sklearn.metrics.f1_score(y_test_vec,y_pred_mat)
    try:
        fpr, tpr, _ = roc_curve(y_test_vec, y_guess_mat[:, 1].squeeze())
    except:
        import pdb; pdb.set_trace()

    roc_auc = auc(fpr, tpr)
    if lamb not in inner_dic.keys():
        inner_dic[lamb] = {}
    inner_dic[lamb]['auc'] = roc_auc
    inner_dic[lamb]['f1'] = f1
    inner_dic[lamb]['loss'] = test_running_loss / (len(TRAIN_L)+1)       
    
    return inner_dic


if __name__ == "__main__":
    # path = '/PHShome/jjd65/CDIFF/cdiff_metabolomics/outdir/'
    parser = argparse.ArgumentParser()
    cd = cdiffDataLoader()
    cd.make_pt_dict(cd.cdiff_raw)

    parser.add_argument("-dtype", "--data_type",
                        help="type of data", type=str)

    parser.add_argument("-lambda", "--lambda_val",
                        help="lambda to test", type=float)
    
    parser.add_argument("-reg", "--regularizer",
                        help="regularizer", type=int)
    
    parser.add_argument("-w", "--weighting",
                        help="weighting", type=bool)
        
    args = parser.parse_args()

    ml = mlMethods(cd.pt_info_dict, lag=1)
    
    data = ml.data_dict[args.data_type]
    labels = ml.targets_int[args.data_type]

    path = 'outputs_june18/'
    import os
    if not os.path.isdir(path):
        os.mkdir(path)
    ml.path = path
    try:
        with open(args.data_type + '_ixs.pkl', 'rb') as f:
            ixx = pickle.load(f)
    except:
        ixx = ml.leave_one_out_cv(data, labels)
        pickle.dump(ixx, open(path + "ixs.pkl", "wb"))

    outer_loops = len(ixx)
    for i in range(outer_loops):
        ix_in = ixx[i]
        ixtrain, ixtest = ix_in

        TRAIN, TRAIN_L, TEST, TEST_L = data.iloc[ixtrain,
                                                :], labels[ixtrain], data.iloc[ixtest, :], labels[ixtest]

        epochs = 1000

        try:
            with open(path + args.data_type + '_' + str(args.weighting) + '_' + str(args.regularizer) + '_inner_dic.pkl', 'rb') as f:
                inner_dic = pickle.load(f)
        except:
            inner_dic = {}
        zip_ixs = ml.leave_one_out_cv(TRAIN, TRAIN_L)
        net = LogRegNet(TRAIN.shape[1])
        optimizer = torch.optim.RMSprop(net.parameters(), lr=.0001)
        inner_dic = main_parallelized(ml, args.lambda_val, zip_ixs, net, optimizer, TRAIN,
                        TRAIN_L, epochs, args.weighting, args.regularizer, inner_dic)

        pickle.dump(inner_dic, open(path + args.data_type + '_' +
                                    str(args.weighting) + '_' + str(args.regularizer) + "inner_dic.pkl", "wb"))
        print('loop ' + str(i) +' Complete')

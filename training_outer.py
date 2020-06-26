import scipy.stats as st
from collections import Counter
from ml_methods import *
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import argparse
import pickle
import os
import numpy as np
# def train_loop(ml, train_data, train_labels, net, optimizer, criterion, lamb_to_test, regularizer):


# def test_loop(ml,net, X_test, y_test, criterion):
#     net.eval()
#     test_out = net(X_test).double()
#     if len(test_out.shape) == 1:
#         test_out = test_out.unsqueeze(0)

#     # find loss
#     test_loss = criterion(
#         test_out, ml.make_one_hot(y_test, 2))

#     m = nn.Softmax(dim=1)
#     test_out_sig = m(test_out)

#     y_guess = test_out_sig.detach().numpy()

#     # find f1 score
#     test_loss = test_loss.detach().numpy().item()
#     return test_out, test_loss, y_guess

# @profile
def train_net(ml, epochs, labels, data, loo_inner = True, loo_outer = True, folds = 3, regularizer = None, weighting = True, lambda_grid=None, train_inner = True, perc = None, ixs = None):
    # Inputs:
        # NNet - net to use (uninitialized)
        # epochs - number of epochs for training inner and outer loop
        # data & labels
        # loo - whether or not to use leave one out cross val
        # folds - number of folds for inner cv
        # shuffle - whether to shuffle 
        # regularizer - either 1 for l1, 2 for l2, or None
        # weighting - either true or false
        # lambda_grid - vector of lambdas to train inner fold over or, if train_inner = False, lambda value for outer CV
        # train_inner - whether or not to train inner fold
        # optimization - what metric ('auc','loss','f1') to use for both early stopping and deciding on lambda value. If loo = True, optimization should be 'loss'
        # perc - Remove metabolites with variance in the bottom 'perc' percent if perc is not None
    
    
    if loo_outer:
        assert(ixs is not None)

    if isinstance(lambda_grid, int):
        train_inner = False

    # Filter variances if perc is not none
    if perc is not None:
        data = ml.filter_vars(data, labels, perc=perc)

    # Split data in outer split
    if not loo_outer:
        ixtrain, ixtest = ml.split_test_train(data, labels)                                
    else:
        ixtrain, ixtest = ixs
    # Normalize data and fix instances where stdev(data) = 0
    dem = np.std(data, 0)
    dz = np.where(dem == 0)[0]
    dem[dz] = 1
    data = (data - np.mean(data, 0))/dem
    
    TRAIN, TRAIN_L, TEST, TEST_L = data.iloc[ixtrain,:], labels[ixtrain], data.iloc[ixtest, :], labels[ixtest]

    # print('Train Recur:' + str(sum(TRAIN_L == 1)))
    # print('Train Cleared:' + str(sum(TRAIN_L == 0)))
    if isinstance(ixtest, int):
        TEST, TEST_L = torch.FloatTensor([np.array(TEST)]), torch.DoubleTensor([[TEST_L]])
    else:
        TEST, TEST_L = torch.FloatTensor(
            np.array(TEST)), torch.DoubleTensor(TEST_L)

    if regularizer is not None:
        reg_param = 'l' + str(regularizer)
    else:
        reg_param = None
    
    if weighting is not None:
        weight_param = 'balanced'
    else:
        weight_param = None

    

    if regularizer is None:
        best_lambda = 0
        lr_mod = sklearn.linear_model.LogisticRegression(
            penalty='none',class_weight=weight_param, max_iter=200, solver = 'newton-cg')
    elif regularizer is not None and not train_inner:
        assert(isinstance(lambda_grid, float))
        best_lambda = lambda_grid
        lr_mod = sklearn.linear_model.LogisticRegression(
            penalty=reg_param , C=1/best_lambda, class_weight=weight_param, solver='liblinear', max_iter=200)
    
    lr_mod.fit(np.array(TRAIN), np.array(TRAIN_L))
    pred_probs_lr = lr_mod.predict_proba(np.array(TEST))
    pred_lr = lr_mod.predict(np.array(TEST))
    print('TRUE: ' + str(TEST_L))
    print('PROBS: ' + str(pred_probs_lr))
    print('GUESS: ' + str(pred_lr))
    # print(ixtrain)
    # print(ixtest)

    
    # initialize net with TRAIN shape
    net = LogRegNet(TRAIN.shape[1])
    net.apply(ml.init_weights)

    optimizer = torch.optim.RMSprop(net.parameters(), lr=.001)
    # optimizer = torch.optim.SGD(net.parameters(), lr=.01, momentum = 0.9)

    
    # Now, train outer loop
    TRAIN = torch.FloatTensor(np.array(TRAIN))
    TRAIN_L = torch.DoubleTensor(np.array(TRAIN_L))
    if weighting:
        weights = len(TRAIN_L) / (2 * np.bincount(TRAIN_L))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights))

    else:
        criterion = nn.BCEWithLogitsLoss()
            
    
    y_guess = []
    test_running_loss = []
    net_vec = []
    net.apply(ml.init_weights)
    loss_vec = []
    y_true = []
    for epoch in range(epochs):
        # out = ml.train_loop(TRAIN, TRAIN_L, net,
        #                         optimizer, criterion, best_lambda, regularizer)

        net.train()
        optimizer.zero_grad()
        out = net(TRAIN).double()

        reg_lambda = best_lambda
        l2_reg = None
        for W in net.parameters():
            if l2_reg is None:
                l2_reg = W.norm(regularizer)
            else:
                l2_reg = l2_reg + W.norm(regularizer)
        if len(out.shape) == 1:
            out = out.unsqueeze(0)

        loss = criterion(out, ml.make_one_hot(
            TRAIN_L, 2)) + reg_lambda * l2_reg

        loss.backward()
        optimizer.step()

        
        # And test outer loop
        net.eval()
        test_out = net(TEST).double()
        loss_vec.append(loss.item())

        # calculate loss
        try:
            test_loss = criterion(test_out, ml.make_one_hot(TEST_L,2))
        except:
            test_out = test_out.unsqueeze(0)
            test_loss = criterion(test_out, ml.make_one_hot(TEST_L, 2))
        mm = nn.Softmax(dim=1)
        test_out_sig = mm(test_out)
        y_guess.append(test_out_sig.detach().numpy())

        test_running_loss.append(test_loss.item())
        net_vec.append(net)

        y_pred = np.argmax(y_guess[-1],1)
        
        running_vec = loss_vec
        # bool_test = np.array([r1 >= r2 for r1, r2 in zip(
        #         running_vec[-10:], running_vec[-11:-1])]).all()
        if len(running_vec) > 3:
            bool_test = np.abs(running_vec[-2] - running_vec[-1]) < 1e-4
        
        
        # y_true.append(TEST_L.detach().numpy())

        
        if epoch > 50 and bool_test:
            print(epoch)
            break
    # plt.figure()
    # plt.plot(test_running_loss, label = 'test_loss')
    # plt.plot(loss_vec, label = 'train_loss')
    # plt.legend()
    # plt.show()

    # print('outputs')
    # print(y_guess[-1])
    # print(TEST_L)

    # record best net & y_guess
    net_out = net_vec[-11]
    y_guess_fin = y_guess[-11]
    y_true = TEST_L

    return y_guess_fin, y_true, net_out, best_lambda, running_vec, pred_lr

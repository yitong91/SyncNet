

import time
import scipy.io
import os
import numpy as np
import theano
import theano.tensor as tensor
import theano.tensor.signal
import theano.tensor.signal.pool

from cnn_classifier import init_params, init_tparams
from cnn_classifier import build_model
import sklearn
from sklearn import metrics
import optimizers

from utils import get_minibatches_idx, Read_Autism_cross
from utils import  unzip, zipp

SEED = 1101
def pred_error(f_pred, data, label):
    preds = f_pred(data)
    errs = (preds == label).sum().astype(theano.config.floatX)
    errs = 1. - errs/data.shape[0]
    errs = errs.astype(theano.config.floatX)
    return errs
    
def pred_error_all(f_pred, f_pred_prob, data, label):
    step = 2000.
    N,C,T = data.shape
    temp = f_pred_prob(data[0:1,:,:])
    pred = np.zeros((data.shape[0],))
    prob = np.zeros((data.shape[0],temp.shape[1]))
    for i in range(np.int32(np.ceil(data.shape[0]/step))):
        srt = np.int32(i * step)
        edn = np.int32(min((i+1) * step, data.shape[0]))
        pred[srt:edn] = f_pred(data[srt:edn,:,:])
        prob[srt:edn,:] = f_pred_prob(data[srt:edn,:,:])
    errs = (pred == label).sum().astype(theano.config.floatX)
    errs = 1. - errs/data.shape[0]
    errs = errs.astype(theano.config.floatX)
    return prob, pred, errs

""" Training the model. """


""" used to calculate the prediction error. """

if __name__ == '__main__':
    # data is of size N x C x T
    # https://docs.python.org/2/howto/logging-cookbook.html
    Nt = 40
    K = 10
    batch_size = 200
    max_epochs = 40
    patience = 10
    lrate = 0.002
    valid_batch = 100
    dispFreq = 2
    validFreq = 10
    pool_size_cl = 40
    C = 60
    drop_rate = 0.2
    saveFreq = 100
    result_path = './Result_gpcnn_p' + str(drop_rate) + '_K_' + str(K) + '_Nt' + str(Nt) + '_C' + str(C) 
    file_path = '/media/lyt/SSD/Autism/All_zscores_200/'
    if not os.path.exists(result_path):
            os.mkdir(result_path)
    for test_id in np.array([21]):
        if test_id == 4 or test_id == 7 or test_id == 22:
            continue
        train,val,test,labtrain,_,labval,labtest = Read_Autism_cross(file_path,[test_id + 1])
        _,C0,T = train.shape
        options = {}
        options['uidx'] = 0
        options['P'] = 2
        options['sigma'] = np.float32(0.001)
        options['C0'] = C0
        mat = scipy.io.loadmat('./Autism_position.mat')
        options['Lx'] = mat['chanlocs'].astype('float32') 
        options['T'] = T
        options['C'] = C
        options['e_K'] = K
        options['e_Nt'] = Nt
        options['e_filter_shape'] = (K,C,1,Nt)
        options['cl_pool_size'] = pool_size_cl
        options['Wy'] =int(np.floor(float(T)/float(options['cl_pool_size'])))
        options['dropout_rate'] = drop_rate
        options['cl_Wy'] = options['Wy'] * K         
        options['cl_ny'] = np.max(labtrain) + 1
        options['pre_Wy'] = options['Wy'] * K
        options['pre_ny'] = options['cl_ny']
        options['n_y'] = np.max(labtrain) + 1
        estop = False
        history_errs = []
        history_aucs = []
        best_p = None
        bad_counter = 0
        uidx = 0 # number of update done
        inits = init_params(options,'all')
        params = inits
        tparams = init_tparams(params)
        before = np.zeros((2,))
        _x,_y,f_conv,f_pred_prob, f_pred, _cost= build_model(tparams,options)
        _lr = tensor.scalar(name = 'lr')
        f_cost = theano.function([_x,_y],_cost)
        f_grad_shared, f_update = optimizers.Adam(tparams,_cost,[_x,_y],_lr)
    
        print('Start Pre-Training...')
        start_time = time.time()    
        try:
            for eidx in xrange(max_epochs):
                batch_index = get_minibatches_idx(train.shape[0], batch_size,shuffle = True)
                for _, train_index in batch_index:
                    uidx = uidx + 1
                    options['uidx'] = options['uidx'] + 1
                    x = train[train_index,:,:]
                    y = labtrain[train_index]
                    cost = f_grad_shared(x,y)
                    f_update(lrate,0.)
                if np.mod(eidx + 1,dispFreq) == 0:
                    print('Epoch ' + str(eidx) + ' Update ' + str(uidx) + ' Cost ' + str(cost))
                if np.mod(eidx + 1, saveFreq) == 0:
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez('./model.npz',history_errs = history_errs,**params)
                if np.mod(eidx + 1,validFreq) == 0:
                    train_prod,train_pred, train_err = pred_error_all(f_pred,f_pred_prob, train, labtrain)
                    #train_auc = metrics.roc_auc_score(labtrain,train_pred[:,1])
                    val_prod, val_pred, val_err = pred_error_all(f_pred,f_pred_prob, val, labval)
                    #val_auc = metrics.roc_auc_score(labval,val_pred[:,1])
                    test_prod, test_pred, test_err = pred_error_all(f_pred,f_pred_prob, test, labtest)
                    #test_auc = metrics.roc_auc_score(labtest,test_pred[:,1])
                    history_errs.append([train_err,val_err,test_err])
                    #history_aucs.append([train_auc,val_auc,test_auc])                
                    print('Train ' + str(train_err) + ' Val ' + str(val_err) + ' Test ' + str(test_err))
                    #print('Train ' + str(train_auc) + ' Val ' + str(val_auc) + ' Test ' + str(test_auc))             
                    if uidx == 0 or val_err<=np.array(history_errs)[:,1].min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                        before[0] = test_err
                        #before[1] = test_auc
                        confusion = sklearn.metrics.confusion_matrix(labtest,test_pred)
                    if len(history_errs) > patience and val_err >= np.array(history_errs)[:-patience,0].min():
                        bad_count = bad_counter + 1
                        if bad_counter > patience:
                            estop = True
                            break
                if estop:
                    break
                    
                    
                    
        except KeyboardInterrupt:
            print('Training interrupted')
            end_time = time.time()
            if best_p is not None:
                zipp(best_p,tparams)
            else:
                best_p = unzip(tparams)
                
                
                
        max_epochs = 100
        patience = 10
        lrate = 0.002
        valid_batch = 100
        print('Start Training...')
        start_time = time.time()    
        try:
            for eidx in xrange(max_epochs):
                batch_index = get_minibatches_idx(train.shape[0], batch_size,shuffle = True)
                for _, train_index in batch_index:
                    uidx = uidx + 1
                    options['uidx'] = options['uidx'] + 1
                    x = train[train_index,:,:]
                    y = labtrain[train_index]
                    cost = f_grad_shared(x,y)
                    f_update(lrate,1.)
                if np.mod(eidx + 1,dispFreq) == 0:
                    print('Epoch ' + str(eidx) + ' Update ' + str(uidx) + ' Cost ' + str(cost))
                if np.mod(eidx + 1, saveFreq) == 0:
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez('./model.npz',history_errs = history_errs,**params)
                if np.mod(eidx + 1,validFreq) == 0:
                    train_prod, train_pred, train_err = pred_error_all(f_pred,f_pred_prob, train, labtrain)
                    #train_auc = metrics.roc_auc_score(labtrain,train_pred[:,1])
                    val_prod, val_pred, val_err = pred_error_all(f_pred,f_pred_prob, val, labval)
                    #val_auc = metrics.roc_auc_score(labval,val_pred[:,1])
                    test_prod, test_pred, test_err = pred_error_all(f_pred,f_pred_prob, test, labtest)
                    #test_auc = metrics.roc_auc_score(labtest,test_pred[:,1])
                    history_errs.append([train_err,val_err,test_err])
                    #history_aucs.append([train_auc,val_auc,test_auc])                
                    print('Train ' + str(train_err) + ' Val ' + str(val_err) + ' Test ' + str(test_err))
                    #print('Train ' + str(train_auc) + ' Val ' + str(val_auc) + ' Test ' + str(test_auc))             
                    if uidx == 0 or val_err<=np.array(history_errs)[:,1].min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                        before[0] = test_err
                        #before[1] = test_auc
                        confusion = sklearn.metrics.confusion_matrix(labtest,test_pred)
                    if len(history_errs) > patience and val_err >= np.array(history_errs)[:-patience,0].min():
                        bad_count = bad_counter + 1
                        if bad_counter > patience:
                            estop = True
                            break
                if estop:
                    break
                    
        except KeyboardInterrupt:
            print('Training interrupted')
            end_time = time.time()
            if best_p is not None:
                zipp(best_p,tparams)
            else:
                best_p = unzip(tparams)
        scipy.io.savemat(result_path + '/Result_' + str(test_id + 1) + '_' + str(before[0]) + '.mat',{'inits':inits, 'best_p':best_p, 'history_errs':history_errs,'confusion':confusion})    
        print('Best Test Error is: ' + str(before[0]) + ', AUC is' + str(before[1]))

import numpy as np
import theano
import os
import scipy.io
from theano import config
from collections import OrderedDict

def Read_Autism_cross(dir_path,key):
    train = []
    val = []
    test = []
    labtrain = []
    labtrain2 = []
    labval = []
    labtest = []
    pool = []
    lab_dic = {}
    lab_dic2 = {}
    for i in range(1,26):
        if i in key:
            continue
        pool.append(str(i).zfill(3))
        lab_dic2[str(i).zfill(3)] = i - 1
    pool = np.array(pool)
    shuffle = np.random.permutation(len(pool))
    N1 = int(np.ceil(0.8 * len(pool)))
    train_pool = pool[shuffle[:N1]]
    val_pool = pool[shuffle[N1:]]
    files = [f for f in os.listdir(dir_path) if f.endswith('.mat')]
    for k in range(len(key)):
        key[k] = str(key[k]).zfill(3)
    lab_dic['T1'] = 0
    lab_dic['T2'] = 1
    lab_dic['T3'] = 2
    for idx in range(len(files)):
        f_id = files[idx][18:21]
        f_lab = files[idx][22:24]       
        mat = scipy.io.loadmat(dir_path + files[idx])
        data = mat['trial'].astype('float32')
        if f_id in train_pool:
            train.append(data.astype('float32'))
            labtrain.append(lab_dic[f_lab] * np.ones((data.shape[0],)))
            labtrain2.append(lab_dic2[f_id] * np.ones((data.shape[0],)))
        elif f_id in val_pool:
            val.append(data.astype('float32'))
            labval.append(lab_dic[f_lab] * np.ones((data.shape[0],)))
        elif f_id in key:
            test.append(data.astype('float32'))
            labtest.append(lab_dic[f_lab] * np.ones((data.shape[0],)))
        else:
            print 'Error in Reading' + f_id
    train = np.concatenate(train,axis = 0)
    val = np.concatenate(val,axis = 0)
    test = np.concatenate(test,axis = 0)
    labtrain = np.concatenate(labtrain,axis = 0).astype('int32')
    labtrain2 = np.concatenate(labtrain2,axis = 0).astype('int32')
    labval = np.concatenate(labval,axis = 0).astype('int32')
    labtest = np.concatenate(labtest,axis = 0).astype('int32')
    return train,val,test,labtrain,labtrain2,labval,labtest



def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)
    
def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

     
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
    
def _p(pp, name):
    return '%s_%s' % (pp, name)

def dropout(X, trng, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

def dropout_channel(X, trng, p=0.):
    if p != 0:
        retain_prob = 1 - p
        C = X.shape[1]
        N = X.shape[0]    
        _n = trng.binomial((N,C), p = retain_prob, dtype = theano.config.floatX).dimshuffle(0,1,'x')
        X = X / retain_prob * _n
    return X
""" used for initialization of the parameters. """

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)
    
def uniform_weight(nin,nout=None, lowscale=-0.05, highscale = 0.05):
    if nout == None:
        nout = nin
    W = np.random.uniform(low=lowscale, high=highscale, size=(nin, nout))
    return W.astype(config.floatX)
    
def normal_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.randn(nin, nout) * scale
    return W.astype(config.floatX)
    
def zero_bias(ndim):
    b = np.zeros((ndim,))
    return b.astype(config.floatX)







 

    

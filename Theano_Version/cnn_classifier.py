
import numpy as np
import theano
import theano.tensor as tensor

from collections import OrderedDict

from utils import uniform_weight, dropout_channel, zero_bias, numpy_floatX
from theano.tensor.nnet import conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from cnn_layers import conv_pool, conv_pool2, deconv_depool, conv_pool_ae, ReLU
from utils import _p

# Set the random number generators' seeds for consistency
SEED = 3435
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

""" init. parameters. """  

def init_params(options,prefix):
    temp = prefix
    params = OrderedDict()
    selected = np.random.permutation(options['C0'])
    params['r'] = options['Lx'][selected[0:options['C']],:]
    params['gp_beta'] = np.float32(1.)
    params['gp_alpha'] = np.float32(2.)
    if prefix == 'e' or temp == 'all':
        prefix = 'e'
        params[_p(prefix,'b')] = uniform_weight(options[_p(prefix,'K')],options['C'])
        params[_p(prefix,'phi')] = uniform_weight(options[_p(prefix,'K')],options['C'] - 1)
        params[_p(prefix,'beta')] = uniform_weight(options[_p(prefix,'K')],1,0,0.05)
        params[_p(prefix,'omega')] = uniform_weight(options[_p(prefix,'K')],1, 0,1.)
        params[_p(prefix,'bias')] = np.zeros((options[_p(prefix,'K')]),dtype = theano.config.floatX)
        
    if prefix == 'cl' or temp == 'all':   
        prefix = 'cl'
        params[_p(prefix,'Wy')] = uniform_weight(options[_p(prefix,'Wy')],options[_p(prefix,'ny')])
        params[_p(prefix,'by')] = np.zeros((options[_p(prefix,'ny')]),dtype = theano.config.floatX)    
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
        #tparams[kk].tag.test_value = params[kk]
    return tparams

def _slice(_x,n,dim):
    if _x.ndim == 3:
        return _x[:,:,n * dim:(n+1)*dim]
    return _x[:, n * dim:(n+1)*dim]
    
def Gaussian_Process(tparams,options):
    #_rzz = tparams['r'].repeat(options['C'],axis = 0).reshape((options['C'],options['C'],3))     
    #_distancezz = (_rzz - _rzz.dimshuffle(1,0,2)).norm(1, axis = 2)
    #_Kzz = tparams['gp_beta'] * tensor.exp( - tparams['gp_alpha'] * _distancezz)          
    _rxx = np.repeat(options['Lx'],options['C0'], axis = 0).reshape((options['C0'],options['C0'],options['P']))
    _distancexx = np.linalg.norm(_rxx - _rxx.transpose(1,0,2), 1, axis = 2)    
    _rzx = tparams['r'].repeat(options['C0'],axis = 0 ).reshape((options['C'],options['C0'],options['P']))
    _Kxx = tparams['gp_beta'] * tensor.exp( - tparams['gp_alpha'] * _distancexx)
    _x_rep = np.repeat(options['Lx'], options['C'], axis = 0).reshape((options['C0'], options['C'] , options['P']))
    _distancezx = (_rzx - _x_rep.transpose((1,0,2))).norm(1,axis = 2)
    _Kzx = tparams['gp_beta'] * tensor.exp( - tparams['gp_alpha'] * _distancezx)
    _eta = tensor.nlinalg.matrix_inverse(_Kxx + options['sigma'] * np.eye(options['C0'],dtype = 'float32'))
    _W = _Kzx.dot(_eta) 
    return _W
    
def inv_Gaussian_Process(tparams,options):
    _rzz = tparams['r'].repeat(options['C'],axis = 0).reshape((options['C'],options['C'],options['P']))     
    _distancezz = (_rzz - _rzz.dimshuffle(1,0,2)).norm(1, axis = 2)
    _Kzz = tparams['gp_beta'] * tensor.exp( - tparams['gp_alpha'] * _distancezz)          
#    _rxx = np.repeat(options['Lx'],options['C0'], axis = 0).reshape((options['C0'],options['C0'],options['P']))
#    _distancexx = np.linalg.norm(_rxx - _rxx.transpose(1,0,2), 1, axis = 2)    
#    _Kxx = tparams['gp_beta'] * tensor.exp( - tparams['gp_alpha'] * _distancexx)
    _rzx = tparams['r'].repeat(options['C0'],axis = 0 ).reshape((options['C'],options['C0'],options['P']))    
    _x_rep = np.repeat(options['Lx'], options['C'], axis = 0).reshape((options['C0'], options['C'] , options['P']))
    _distancezx = (_rzx - _x_rep.transpose((1,0,2))).norm(1,axis = 2)
    _Kzx = tparams['gp_beta'] * tensor.exp( - tparams['gp_alpha'] * _distancezx)
    _Kxz = _Kzx.dimshuffle(1,0)
    _eta = tensor.nlinalg.matrix_inverse(_Kzz + options['sigma'] * np.eye(options['C'],dtype = 'float32'))
    _W = _Kxz.dot(_eta) 
    return _W
""" Building model... """

def build_model(tparams,options):
    
    # Used for dropout.
    #use_noise = theano.shared(numpy_floatX(0.))
    # input sentence: n_samples * n_steps 
    _x = tensor.tensor3('x', dtype=theano.config.floatX)
    _y = tensor.vector('y',dtype='int32')
    trng = RandomStreams(10)
    p = theano.shared(np.float32(options['dropout_rate']))    
    _x_dropout = dropout_channel(_x,trng,p)    
    
    _Wzx = Gaussian_Process(tparams,options)
    _Wzx = _Wzx.dimshuffle(1,0)
    _input = tensor.dot(_x_dropout.dimshuffle(0,2,1),_Wzx)
    _input = _input.dimshuffle(0,2,'x',1)
    
    def encoder(layer0_input):
        
        """ filter_shape: (number of filters, num input feature maps, filter height,
                            filter width)
            image_shape: (batch_size, num input feature maps, image height, image width)
        """
        output = conv_pool(layer0_input, tparams, options, 'e')        
        output = output.reshape((output.shape[0],options['cl_Wy'])) 
               
        
        return output
    
    _output = encoder(_input)    
    f_conv = theano.function([_x], _output)    
    # this is the label prediction you made 
    pred = tensor.nnet.softmax(tensor.dot(_output, tparams['cl_Wy']) + tparams['cl_by'])   
    f_pred_prob = theano.function([_x], pred, name='f_pred_prob')
    f_pred = theano.function([_x], pred.argmax(axis=1), name='f_pred')

    index = tensor.arange(_x.shape[0])
    cost = -tensor.log(pred[index, _y] + 1e-6).mean() #+ 10 * theano.tensor.abs_(tparams['b']).mean() #+10**2 *  theano.tensor.abs_(tparams['phi']).mean()
  
    return _x,_y,f_conv, f_pred_prob,f_pred,cost


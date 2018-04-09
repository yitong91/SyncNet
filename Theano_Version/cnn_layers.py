
import numpy as np
import theano
import theano.tensor as tensor
import theano.tensor.shared_randomstreams
from theano.tensor.nnet import conv
from utils import _p

rng = np.random.RandomState(3435)

def ReLU(x):
    y = theano.tensor.maximum(0.0, x)
    return(y)
    
def get_filter(tparams,options,prefix):
    if prefix is 'e':
        Channel_num = options['C']
        Filter_num = options['e_K']
        Filter_len = options['e_Nt']
    elif prefix is 'd':
        Channel_num = options['e_K']
        Filter_num = options['C']
        Filter_len = options['e_Nt']
    elif prefix is 'e2':
        Channel_num = options['e_K']
        Filter_num = options['e2_K']
        Filter_len = options['e2_Nt']
        
    zero_pad = tensor.basic.zeros([Filter_num,1])
    #_beta = (tparams['beta'] + theano.tensor.abs_(tparams['beta']))/2
    _beta = tensor.extra_ops.repeat(tparams[_p(prefix,'beta')],Channel_num,axis = 1).dimshuffle(0,1,'x','x')    
    _beta = _beta.repeat(Filter_len,axis = 3)
    _omega = tensor.extra_ops.repeat(tparams[_p(prefix,'omega')],Channel_num,axis = 1).dimshuffle(0,1,'x','x')
    _omega = _omega.repeat(Filter_len,axis = 3)    
    _b = tparams[_p(prefix,'b')].dimshuffle(0,1,'x','x')
    _b = _b.repeat(Filter_len,axis = 3)
    _phi = tensor.concatenate([zero_pad,tparams[_p(prefix,'phi')]],axis = 1).dimshuffle(0,1,'x','x')
    _phi = _phi.repeat(Filter_len,axis = 3)
    t = np.array(range(-Filter_len/2,Filter_len/2)).reshape(1,1,1,-1)
    t = t.repeat(Filter_num,axis = 0) 
    t = t.repeat(Channel_num,axis = 1)
    _W = _b * tensor.cos(_omega * t + _phi) * tensor.exp(-_beta * t * t)    
    return _W
     

def depool_repeat(_X,rate):
    _X_repeat = theano.tensor.extra_ops.repeat(_X.dimshuffle(0,1,'x',2),rate,axis = 3)
    _output = theano.tensor.extra_ops.squeeze(_X_repeat)
    return _output
""" Encoder using Convolutional Neural Network. """    


    

def conv_pool_ae(layer0_input, tparams, options, prefix):    
    """ filter_shape: (number of filters, num input feature maps, filter height,
                        filter width)
        image_shape: (batch_size, num input feature maps, image height, image width)
    """
    _W = get_filter(tparams,options,prefix).astype(theano.config.floatX)
    s = int(np.floor(options[_p(prefix,'Nt')]/2.))
    conv_out = conv.conv2d(input=layer0_input, filters=_W, 
                            filter_shape= options[_p(prefix,'filter_shape')], border_mode = 'full')[:,:,:,s-1:-s]
    conv_out_relu = ReLU(conv_out + tparams[_p(prefix,'bias')].dimshuffle('x', 0, 'x', 'x'))
    output_re =  theano.tensor.signal.pool.pool_2d(input=conv_out_relu.dimshuffle(0,2,1,3),
                                                ds=(1,options[_p(prefix,'pool_size')]), ignore_border=True, mode = 'max')    
    output_re =output_re.reshape((output_re.shape[0],options[_p(prefix,'K')],output_re.shape[3]))
    output_cl = theano.tensor.signal.pool.pool_2d(input=conv_out_relu.dimshuffle(0,2,1,3),
                    ds=(1,options['cl_pool_size']), ignore_border=True, mode = 'max')    
    output_cl = output_cl.reshape((output_cl.shape[0], options[_p(prefix,'K')],output_cl.shape[3]))
   
    return output_cl, output_re


def conv_pool(layer0_input, tparams, options, prefix):    
    """ filter_shape: (number of filters, num input feature maps, filter height,
                        filter width)
        image_shape: (batch_size, num input feature maps, image height, image width)
    """
    _W = get_filter(tparams,options,prefix).astype(theano.config.floatX)
    s = int(np.floor(options[_p(prefix,'Nt')]/2.))
    conv_out = conv.conv2d(input=layer0_input, filters=_W, 
                            filter_shape= options[_p(prefix,'filter_shape')], border_mode = 'full')[:,:,:,s-1:-s]
    conv_out_relu = ReLU(conv_out + tparams[_p(prefix,'bias')].dimshuffle('x', 0, 'x', 'x'))
    output =  theano.tensor.signal.pool.pool_2d(input=conv_out_relu.dimshuffle(0,2,1,3),
                                                ds=(1,options['cl_pool_size']), ignore_border=True, mode = 'max')    
    output = output.reshape((output.shape[0],options[_p(prefix,'K')],output.shape[3]))
    return output
    
def conv_pool2(layer0_input, tparams, options,prefix):
    
    s = max( int(np.floor(options[_p(prefix,'Nt')]/2.)),1)
    h = options[_p(prefix,'filter_shape')][2] - 1
    
    conv_out = theano.tensor.nnet.conv.conv2d(input=layer0_input, filters=tparams[_p(prefix,'W')], 
                            filter_shape= options[_p(prefix,'filter_shape')], border_mode = 'full')[:,:,h,s-1:-s]
    conv_out_relu = ReLU(conv_out + tparams[_p(prefix,'bias')].dimshuffle('x', 0, 'x'))
    output =  theano.tensor.signal.pool.pool_2d(input=conv_out_relu.dimshuffle(0,1,'x',2),
                                                ds=(1,options[_p(prefix,'pool_size')]), ignore_border=True, mode = 'max')    
    output =output.reshape((output.shape[0],options[_p(prefix,'K')],output.shape[3]))
    
    return output
    
def deconv_depool(layer0_input,tparams, options,prefix):
    if prefix == 'd':
        depool_out = depool_repeat(layer0_input,options['e_pool_size'])
    elif prefix == 'd2':
        depool_out = depool_repeat(layer0_input,options['e2_pool_size'])
    s = int(np.floor(options[_p(prefix,'Nt')]/2.))
    _W = get_filter(tparams,options,prefix).astype(theano.config.floatX)
    deconv_out = conv.conv2d(input=depool_out.dimshuffle(0,1,'x',2), filters=_W, 
                            filter_shape= options[_p(prefix,'filter_shape')], border_mode = 'full')[:,:,:,s-1:-s]
    doutput = (deconv_out + tparams[_p(prefix,'bias')].dimshuffle('x',0,'x','x')).reshape((deconv_out.shape[0],options['C'],options['T']))
    return doutput
    
def deconv_depool2(layer0_input,tparams, options,prefix):
    if prefix == 'd':
        depool_out = depool_repeat(layer0_input,options['e_pool_size'])
    elif prefix == 'd2':
        depool_out = depool_repeat(layer0_input,options['e2_pool_size'])
    s = int(np.floor(options[_p(prefix,'Nt')]/2.))
    h = int((2 * options[_p(prefix,'K')] - 2)/2.)
    deconv_out = conv.conv2d(input=depool_out.dimshuffle(0,'x',1,2), filters=tparams[_p(prefix,'W')], 
                            filter_shape= options[_p(prefix,'filter_shape')], border_mode = 'full')[:,:,h,s-1:-s]
    doutput = (deconv_out + tparams[_p(prefix,'bias')].dimshuffle('x',0,'x'))#.reshape((deconv_out.shape[0],options['C'],options['T']))
    return doutput
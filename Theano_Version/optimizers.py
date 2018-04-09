import theano
import theano.tensor as tensor
from utils import numpy_floatX
import numpy

def SGD(tparams, cost, inps, lr,clip_norm=5):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):       
        updated_p = p - lr * g
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 

def Momentum(tparams, cost, inps, lr, momentum=0.9,clip_norm=5):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup) 
    
    updates = []

    for p, g in zip(tparams.values(), gshared): 
        m = theano.shared(p.get_value() * 0.)
        m_new = momentum * m - lr * g
        updates.append((m, m_new))        
        
        updated_p = p + m_new
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 

def NAG(tparams, cost, inps, lr, momentum=0.9,clip_norm=5):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup) 
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        m_new = momentum * m - lr * g
        updates.append((m, m_new))        
        
        updated_p = p + momentum * m_new - lr * g
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 
          
def Adagrad(tparams, cost, inps, lr, epsilon=1e-6,clip_norm=5):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)    
    
    updates = []
    
    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_t = acc + g ** 2
        updates.append((acc, acc_t))
        p_t = p - (lr / tensor.sqrt(acc_t + epsilon)) * g
        updates.append((p, p_t))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 

def Adadelta(tparams, cost, inps, lr, rho=0.95, epsilon=1e-6,clip_norm=5):
    """ default: lr=0.5 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_delta = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc,acc_new)) 
        
        update = g * tensor.sqrt(acc_delta + epsilon) / tensor.sqrt(acc_new + epsilon)
        updated_p = p - lr * update
        updates.append((p, updated_p))
        
        acc_delta_new = rho * acc_delta + (1 - rho) * update ** 2
        updates.append((acc_delta,acc_delta_new))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 


def RMSprop_v1(tparams, cost, inps, lr, rho=0.9, epsilon=1e-6,clip_norm=5):
    """ default: lr=0.001 
        This is the implementation of the RMSprop algorithm used in
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        updated_p = p - lr * (g / tensor.sqrt(acc_new + epsilon))
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update
        
def RMSprop_v2(tparams, cost, inps, lr, rho=0.95, momentum=0.9, epsilon=1e-4, clip_norm=5):
    """ default: lr=0.0001 
        This is the implementation of the RMSprop algorithm used in
        http://arxiv.org/pdf/1308.0850v5.pdf
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)    
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc2 = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1.-rho) * g
        acc2_new = rho * acc + (1.-rho) * (g ** 2)
        updates.append((acc, acc_new))
        updates.append((acc2, acc2_new))
        
        updir = theano.shared(p.get_value() * 0.)
        updir_new = momentum * updir - lr * g / tensor.sqrt(acc2_new -acc_new ** 2 + epsilon)
        updates.append((updir, updir_new))
        
        updated_p = p + updir_new
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 
      
def Adam(tparams, cost, inps, lr, b1=0.1, b2=0.001, e=1e-8, clip_norm=5):
    """ default: lr=0.0002 
        This is the implementation of the Adam algorithm
        Reference: http://arxiv.org/pdf/1412.6980v8.pdf
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/(norm + e) for g in grads]
    zero = numpy.float32(0)
    gshared = [theano.shared(p.get_value() * zero, name='%s_grad'%k) 
                for k, p in tparams.iteritems()]

    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    updates = []

    i = theano.shared(numpy_floatX(0.))    
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr * (tensor.sqrt(fix2) / fix1)
    _s = tensor.scalar('s',dtype = 'float32')
    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * zero)
        v = theano.shared(p.get_value() * zero)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        if tensor.eq(_s,0.) and (p.name is 'gp_beta' or p.name is 'gp_alpha' or p.name is 'r'):
            p_t = p - (_s * lr_t * g_t)
        #elif tensor.eq(_s,1.) and (p.name is not 'gp_beta' and p.name is not 'gp_alpha' and  p.name is not 'r'):
        #    p_t = p - ((1-_s) * lr_t * g_t)
        if p.name == 'e_beta' or p.name == 'd_beta':
            p_t = p_t * (p_t>0)
        elif p.name is 'gp_beta' or p.name is 'gp_alpha':
            m_t = m_t.astype('float32')
            v_t = v_t.astype('float32')
            p_t = p_t.astype('float32')
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    
    f_update = theano.function([lr,_s], [], updates=updates)
    
    return f_grad_shared, f_update

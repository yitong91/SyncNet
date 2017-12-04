import tensorflow as tf
import numpy as np
from SyncNetModel import SyncNetModel
import scipy.io
import os
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4) 


batch_size = 10
num_steps = 2000
valid_steps = 1000
mat = scipy.io.loadmat('toy.mat')
train = mat['train'].astype('float32')
val = mat['val'].astype('float32')
test = mat['test'].astype('float32')
labtrain = utils.to_one_hot(mat['labtrain'])
labval = utils.to_one_hot(mat['labval'])
labtest = utils.to_one_hot(mat['labtest'])
train = train.transpose(0,2,1)
val = val.transpose(0,2,1)
test = test.transpose(0,2,1)
options = {}
options['sample_shape'] = (train.shape[1],train.shape[2])
options['num_labels'] = labtrain.shape[1]
options['batch_size'] = batch_size
options['C'] = train.shape[2]
options['T'] = train.shape[1]
options['K'] = 1
options['Nt'] = 40
options['pool_size'] = 40
options['dropout_rate'] = 0.0
options['cl_Wy'] = int(np.ceil(float(options['T'])/float(options['pool_size'])) * options['K'])
tf.reset_default_graph()  
graph = tf.get_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4) 

model = SyncNetModel(options)
sess =  tf.Session(graph = graph, config=tf.ConfigProto(gpu_options=gpu_options)) 
tf.global_variables_initializer().run(session = sess)
gen_batches = utils.batch_generator([train, labtrain], options['batch_size'])

print('Training...')
for i in range(1, num_steps + 1):          
    p = float(i) / num_steps
    #lr = 0.002 #/ (1. + 10 * p)**0.75
    lr = 0.002
    X, y = gen_batches.next()
    _, batch_loss, y_acc = \
        sess.run([model.train_ops, model.y_loss, model.y_acc],
                 feed_dict={model.X: X, model.y: y, model.lr: lr})
    _ = sess.run(model.beta_op, feed_dict = {})
    if i % 100 == 0:
        print 'iter %d  loss: %f   p_acc: %f  lr: %f' % \
                (i, batch_loss, y_acc, lr)
        
     
    if i % valid_steps == 0:
        train_pred, train_acc = sess.run([model.y_pred, model.y_acc], feed_dict = {model.X: train, model.y:labtrain})
        
        val_pred, val_acc = sess.run([model.y_pred, model.y_acc], feed_dict = {model.X: val, model.y:labval})
        
        test_pred, test_acc = sess.run([model.y_pred, model.y_acc], feed_dict = {model.X: test, model.y:labtest})
        
        print 'train: %.4f  valid: %.4f  test: %.4f ' % \
                (train_acc, val_acc, test_acc)
params = utils.get_params(sess)
np.save('./result.npy',params)
#result = utils.get_params(sess)  
#result_for_save = {}
#keys = result.keys()
#for i in range(len(keys)):
#    temp = keys[i].split('/')
#    temp = temp[1]
#    temp = temp[:-2]
#    result_for_save[temp] = result[keys[i]]
#scipy.io.savemat('./result.mat',{'b':result_for_save['b'], 'phi':result_for_save['phi']})
#    
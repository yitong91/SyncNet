import tensorflow as tf
import utils
import numpy as np

class SyncNetModel(object):
    def __init__(self, options):
        self.l = tf.placeholder(tf.float32, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.sample_type = tf.float32
        self.num_labels = options['num_labels']
        self.sample_shape = options['sample_shape']
        self.batch_size = options['batch_size']
        self.dropout_rate = options['dropout_rate']
        self.X = tf.placeholder(tf.as_dtype(self.sample_type), [None] + list(self.sample_shape), name="input_X")
        self.y = tf.placeholder(tf.float32, [None, self.num_labels], name="input_labels")
        self.train = tf.placeholder(tf.bool, [], name = 'train')
        self._build_model(options)
        self._setup_train_ops()
    
    
    def SyncNetFilters(self, options):
        b=tf.Variable(tf.random_uniform([1,1,options['C'],options['K']], minval=-0.05, maxval=0.05, dtype=tf.float32),name='b')
        omega=tf.Variable(tf.random_uniform ([1,1,1,options['K']], minval = 0., maxval = 1.),name='omega')
        zero_pad = tf.zeros( (1, 1, 1, options['K']), dtype = tf.float32, name ='zero_pad')
        phi_ini=tf.Variable(tf.random_normal([1,1,options['C']-1, options['K']],mean=0.0, stddev=0.05, dtype=tf.float32), name='phi')
        phi = tf.concat([zero_pad, phi_ini], axis = 2)
        beta=tf.Variable(tf.random_uniform([1,1,1,options['K']], minval = 0., maxval = 0.05), dtype = tf.float32,name='beta')
        #t=np.reshape(np.linspace(-options['Nt']/2.,options['Nt']/2.,options['Nt']),[1,options['Nt'],1,1])
        t=np.reshape(range(-options['Nt']/2,options['Nt']/2),[1,options['Nt'],1,1])
        tc=tf.constant(np.single(t),name='t')
        W_osc=tf.multiply(b,tf.cos(tc*omega+phi))
        W_decay=tf.exp(-tf.pow(tc,2)*beta)
        W=tf.multiply(W_osc,W_decay)
        self.beta_op = tf.assign(beta, tf.clip_by_value(beta, 0, np.infty))
        return W

    def feature_extractor(self, X, options):
        self.dropout_x = utils.channel_dropout(X, self.dropout_rate)
        X = tf.expand_dims(self.dropout_x, axis = 1, name = 'reshaped_input')
        with tf.variable_scope('syncnet_conv',reuse = True):
            W = self.SyncNetFilters(options)       
            bias = tf.Variable(tf.constant(0.0, dtype = tf.float32, shape = [options['K']]), name = 'bias')
            h_conv1 = tf.nn.relu(tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME') + bias)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, options['pool_size'], 1], strides=[1, 1, options['pool_size'], 1], padding='SAME')            
            self.h_pool1 = h_pool1
            features = tf.reshape(h_pool1, [-1, options['cl_Wy']])          
        return features
            
    def label_predictor(self, features):
        with tf.variable_scope('label_predictor_logits'):
            logits = utils.fully_connected_layer(features, self.num_labels)    
        return logits
         
    def _build_model(self, options):     
        self.features = self.feature_extractor(self.X, options)        
        logits = self.label_predictor(self.features)
        self.y_pred = tf.nn.softmax(logits)
        self.y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.y))
        self.y_acc = utils.predictor_accuracy(self.y_pred,self.y)
        
        
    def _setup_train_ops(self):
        self.train_ops = tf.train.AdamOptimizer(self.lr).minimize(self.y_loss)
       
        
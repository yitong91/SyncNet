import tensorflow as tf
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy
# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts

def get_params(sess):
    variables = tf.trainable_variables()
    params = {}
    for i in range(len(variables)):
        name = variables[i].name
        params[name] = sess.run(variables[i])
    return params
    
    
def to_one_hot(x, N = -1):
    x = x.astype('int32')
    if np.min(x) !=0 and N == -1:
        x = x - np.min(x)
    x = x.reshape(-1)
    if N == -1:
        N = np.max(x) + 1
    label = np.zeros((x.shape[0],N))
    idx = range(x.shape[0])
    label[idx,x] = 1
    return label.astype('float32')
    
def image_mean(x):
    x_mean = x.mean((0, 1, 2))
    return x_mean

def shape(tensor):
    """
    Get the shape of a tensor. This is a compile-time operation,
    meaning that it runs when building the graph, not running it.
    This means that it cannot know the shape of any placeholders
    or variables with shape determined by feed_dict.
    """
    return tuple([d.value for d in tensor.get_shape()])


def fully_connected_layer(in_tensor, out_units):
    """
    Add a fully connected layer to the default graph, taking as input `in_tensor`, and
    creating a hidden layer of `out_units` neurons. This should be done in a new variable
    scope. Creates variables W and b, and computes activation_function(in * W + b).
    """
    _, num_features = shape(in_tensor)
    weights = tf.get_variable(name = "weights", shape = [num_features, out_units], initializer = tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable( name = "biases", shape = [out_units], initializer=tf.constant_initializer(0.1))
    return tf.matmul(in_tensor, weights) + biases


def conv2d(in_tensor, filter_shape, out_channels):
    """
    Creates a conv2d layer. The input image (whish should already be shaped like an image,
    a 4D tensor [N, W, H, C]) is convolved with `out_channels` filters, each with shape
    `filter_shape` (a width and height). The ReLU activation function is used on the
    output of the convolution.
    """
    _, _, _, channels = shape(in_tensor)
    W_shape = filter_shape + [channels, out_channels]

    # create variables
    weights = tf.get_variable(name = "weights", shape = W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable(name = "biases", shape = [out_channels], initializer= tf.constant_initializer(0.1))
    conv = tf.nn.conv2d( in_tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
    h_conv = conv + biases
    return h_conv


#def conv1d(in_tensor, filter_shape, out_channels):
#    _, _, channels = shape(in_tensor)
#    W_shape = [filter_shape, channels, out_channels]
#    
#    W = tf.truncated_normal(W_shape, dtype = tf.float32, stddev = 0.1)
#    weights = tf.Variable(W, name = "weights")
#    b = tf.truncated_normal([out_channels], dtype = tf.float32, stddev = 0.1)
#    biases = tf.Variable(b, name = "biases")
#    conv = tf.nn.conv1d(in_tensor, weights, stride=1, padding='SAME')
#    h_conv = conv + biases
#    return h_conv

def vars_from_scopes(scopes):
    """
    Returns list of all variables from all listed scopes. Operates within the current scope,
    so if current scope is "scope1", then passing in ["weights", "biases"] will find
    all variables in scopes "scope1/weights" and "scope1/biases".
    """
    current_scope = tf.get_variable_scope().name
    #print(current_scope)
    if current_scope != '':
        scopes = [current_scope + '/' + scope for scope in scopes]
    var = []
    for scope in scopes:
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            var.append(v)
    return var

def tfvar2str(tf_vars):
    names = []
    for i in range(len(tf_vars)):
        names.append(tf_vars[i].name)
    return names


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
        
    
    

def predictor_accuracy(predictions, labels):
    """
    Returns a number in [0, 1] indicating the percentage of `labels` predicted
    correctly (i.e., assigned max logit) by `predictions`.
    """
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)),tf.float32))

def dic2list(sources, targets):
    names_dic = {}
    for key in sources:
        names_dic[sources[key]] = key
    for key in targets:
        names_dic[targets[key]] = key
    names = []
    for i in range(len(names_dic)):
        names.append(names_dic[i])
    return names

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def norm_matrix(X, l):
    Y = np.zeros(X.shape);
    for i in range(X.shape[0]):
        Y[i] = X[i]/np.linalg.norm(X[i],l)
    return Y


def description(sources, targets):
    source_names = sources.keys()
    target_names = targets.keys()
    N = min(len(source_names), 4)
    description = source_names[0]   
    for i in range(1,N):
        description = description  + '_' + source_names[i]
    description = description + '-' + target_names[0]
    return description

def channel_dropout(X, p):
    if p == 0:
        return X
    mask = tf.random_uniform(shape = [tf.shape(X)[0], tf.shape(X)[2]])
    mask = mask + 1 - p
    mask = tf.floor(mask)
    dropout = tf.expand_dims(mask,axis = 1) * X/(1-p)
    return dropout
    
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
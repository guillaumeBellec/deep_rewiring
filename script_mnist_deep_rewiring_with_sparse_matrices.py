import tensorflow as tf
import numpy as np
import numpy.random as rd
from tensorflow.examples.tutorials.mnist import input_data
import time
import pickle

mnist = input_data.read_data_sets("../datasets/MNIST", one_hot=True)

# Define the main hyper parameter accessible from the shell
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('n_epochs', 10, 'number of iteration (55000 is one epoch)')
tf.app.flags.DEFINE_integer('batch', 10, 'number of iteration (55000 is one epoch)')
tf.app.flags.DEFINE_integer('print_every', 100, 'print every k steps')
tf.app.flags.DEFINE_integer('n1', 300, 'Number of neurons in the first hidden layer')
tf.app.flags.DEFINE_integer('n2', 100, 'Number of neurons in the second hidden layer')
#
tf.app.flags.DEFINE_float('p01', .01, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('p02', .03, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('p0out', .3, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('l1', 1e-5, 'l1 regularization coefficient')
tf.app.flags.DEFINE_float('gdnoise', 1e-5, 'gradient noise coefficient')
tf.app.flags.DEFINE_float('lr', 0.5, 'Learning rate')

# Define useful constants
dtype = tf.float32
n_pixels = 28 * 28
n_1 = FLAGS.n1
n_2 = FLAGS.n2
n_out = 10
n_image_per_epoch = mnist.train.images.shape[0]
n_iter = FLAGS.n_epochs * n_image_per_epoch // FLAGS.batch

print_every = FLAGS.print_every
n_minibatch = FLAGS.batch
lr = FLAGS.lr

# Define the number of neurons per layer
sparsity_list = [FLAGS.p01, FLAGS.p02, FLAGS.p0out]
nb_non_zero_coeff_list = [n_pixels * n_1 * FLAGS.p01, n_1 * n_2 * FLAGS.p02, n_2 * n_out * FLAGS.p0out]
nb_non_zero_coeff_list = [int(n) for n in nb_non_zero_coeff_list]

# Placeholders
x = tf.placeholder(dtype, [None, n_pixels])
y = tf.placeholder(dtype, [None, n_out])

def weight_sampler_strict_number(n_in, n_out, nb_non_zero, dtype=tf.float32):
    '''
    Returns a weight matrix and its underlying, variables, and sign matrices needed for rewiring.

    :param n_in:
    :param n_out:
    :param p0:
    :param dtype:
    :return:
    '''
    with tf.name_scope('SynapticSampler'):
        w0_vals = rd.randn(nb_non_zero) / np.sqrt(n_in) # initial weight values

        # Gererate the random mask
        ind_in = tf.random_uniform(shape=[nb_non_zero],maxval=n_in,dtype=tf.int64)
        ind_out = tf.random_uniform(shape=[nb_non_zero],maxval=n_out,dtype=tf.int64)
        indices = tf.stack([ind_in,ind_out],axis=1)
        indices = tf.Variable(initial_value=indices,trainable=False,dtype=tf.int64)

        # Generate random signs
        sign_0 = np.sign(rd.randn(nb_non_zero))

        # Define the tensorflow matrices
        th = tf.Variable(np.abs(w0_vals), dtype=dtype, name='theta')
        w_sign = tf.Variable(sign_0, dtype=dtype, trainable=False, name='sign')
        w = tf.SparseTensor(indices=indices,values=th * w_sign, dense_shape=[n_in,n_out])
        w = tf.sparse_reorder(w)

        return w,w_sign,indices,th

def rewiring(theta, sign, indices,w):
    '''
    The rewiring operation to use after each iteration.
    :param theta:
    :param target_nb_connection:
    :return:

    '''

    with tf.name_scope('rewiring'):
        th = theta.read_value()
        n_in,n_out = w.get_shape()
        target_nb_connection = tf.size(theta)

        #
        reconnect_mask = tf.less_equal(th,0)

        # Gererate the random mask
        ind_in = tf.random_uniform(shape=[target_nb_connection],maxval=n_in,dtype=tf.int64)
        ind_out = tf.random_uniform(shape=[target_nb_connection],maxval=n_out,dtype=tf.int64)
        new_indices = tf.stack([ind_in,ind_out],axis=1)
        new_indices = tf.where(reconnect_mask,new_indices,indices)
        set_indices = tf.assign(indices,new_indices)

        # Generate random signs
        new_sign = tf.random_uniform(shape=[target_nb_connection],maxval=2,dtype=tf.int32) * 2 -1
        new_sign = tf.cast(new_sign,dtype=tf.float32)
        new_sign = tf.where(reconnect_mask, new_sign, sign)
        set_sign = tf.assign(sign, new_sign)

        #new_th = tf.zeros(shape=[target_nb_connection],dtype=tf.float32)
        new_th = tf.random_normal(shape=[target_nb_connection],dtype=tf.float32,stddev=0)
        new_th = tf.where(reconnect_mask,new_th,th)
        set_th = tf.assign(theta,new_th)

        #
        with tf.control_dependencies([set_indices,set_sign,set_th]):
            return tf.no_op('Rewiring')

# Define the computational graph
with tf.name_scope('Layer1'):
    W_1, w_sign_1, w_indices_1, th_1 = weight_sampler_strict_number(n_pixels, n_1, nb_non_zero_coeff_list[0])
    a_1 = tf.sparse_tensor_dense_matmul(W_1, x, adjoint_a=True,adjoint_b=True)
    z_1 = tf.nn.relu(a_1)

with tf.name_scope('Layer2'):
    W_2, w_sign_2, w_indices_2, th_2 = weight_sampler_strict_number(n_1, n_2, nb_non_zero_coeff_list[1])
    a_2 = tf.sparse_tensor_dense_matmul(W_2, z_1, adjoint_a=True,adjoint_b=False)
    z_2 = tf.nn.relu(a_2)

with tf.name_scope('OutLayer'):
    w_out, w_sign_out, w_indices_out, th_out = weight_sampler_strict_number(n_2, n_out, nb_non_zero_coeff_list[2])
    logits_predict = tf.sparse_tensor_dense_matmul(w_out, z_2, adjoint_a=True,adjoint_b=False)
    logits_predict = tf.transpose(logits_predict)

# Make list of weight for convenience
theta_list = [th_1, th_2, th_out]
sign_list = [w_sign_1, w_sign_2, w_sign_out]
index_list = [w_indices_1, w_indices_2, w_indices_out]
weight_list = [W_1, W_2, w_out]

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_predict))
    is_correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=dtype))

# Define the training step operation
with tf.name_scope('Training'):
    apply_gradients = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy, var_list=theta_list)

    mask_connected = lambda th: tf.cast(tf.greater(th, 0), tf.float32)
    noise_update = lambda th: tf.random_normal(stddev=FLAGS.gdnoise, shape=tf.shape(th))

    add_gradient_op_list = [tf.assign_add(th, lr * mask_connected(th) * noise_update(th)) for th in theta_list]
    apply_l1_reg = [tf.assign_add(th, - lr * mask_connected(th) * FLAGS.l1) for th in theta_list]

    with tf.control_dependencies([apply_gradients] + add_gradient_op_list + apply_l1_reg):
        rewire_list = [rewiring(theta, sign, indices, w) for theta, sign, indices, w in
                       zip(theta_list, sign_list, index_list, weight_list)]
        with tf.control_dependencies(rewire_list):
            train_step = tf.no_op('Train')

# Some statistics for monitoring the simulation
with tf.name_scope('Stats'):
    get_size_sp = lambda w: int(w.get_shape()[0]) * int(w.get_shape()[1])
    get_non_zeros_sp = lambda w: tf.reduce_sum(tf.cast(tf.not_equal(w.values,0),dtype=tf.int32))

    nb_entries = [get_non_zeros_sp(w) for w in weight_list]
    sizes = [get_size_sp(w) for w in weight_list]

    layer_connectivity = [get_non_zeros_sp(w) / get_size_sp(w) for w in weight_list]
    global_connectivity = tf.reduce_sum(nb_entries) / tf.reduce_sum(sizes)

#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

results = {
    'loss_list': [],
    'acc_list': [],
    'global_connectivity_list': [],
    'layer_connectivity_list': [],
    'n_synapse': [],
    'iteration_list': [],
    'epoch_list': [],
    'turnover_list': [],
    'training_time_list': []}


turnover = [0,0,0]
training_time = 0
testing_time = 0
acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
for k in range(n_iter):

    layer_connectivity_numpy = sess.run(layer_connectivity)
    global_connectivity_numpy = sess.run(global_connectivity)

    if np.mod(k, print_every) == 0:
        t0 = time.time()
        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
        testing_time = time.time() - t0

        print('Epoch: {} \t time/it: {:.2g} s \t time/test: {:.2g} s  \t it: {} \t acc: {:.2g} \t loss {:.2g} \t sparsity: {:.2g} \t layer wise:'.format(
                mnist.train.epochs_completed, training_time, testing_time, k, acc, loss, global_connectivity_numpy) + np.array_str(
                np.array(layer_connectivity_numpy), precision=2))

    for key, variable in zip(
            ['loss', 'acc', 'global_connectivity', 'layer_connectivity', 'iteration', 'epoch', 'training_time',
             'turnover'],
            [loss, acc, global_connectivity_numpy, layer_connectivity_numpy, k, mnist.train.epochs_completed,
             training_time, turnover]):
        results[key + '_list'].append(variable)

    if np.mod(k, print_every) == 0:
        th_np_old = sess.run(theta_list)

    batch_xs, batch_ys = mnist.train.next_batch(n_minibatch)
    t0 = time.time()
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    training_time = time.time() - t0

    if np.mod(k, print_every) == 0:
        th_np_new = sess.run(theta_list)
        creation_nbs = [np.sum(np.logical_and(w_new > 0, w_old <= 0)) for w_new, w_old in zip(th_np_new, th_np_old)]
        deletion_nbs = [np.sum(np.logical_and(w_new <= 0, w_old > 0)) for w_new, w_old in zip(th_np_new, th_np_old)]
        n_cons = [np.sum(w_new > 0) for w_new in th_np_new]

        turnover = creation_nbs
        print('Syn created: {} {} {}'.format(creation_nbs[0], creation_nbs[1], creation_nbs[2]))
        print('Syn deleted: {} {} {}'.format(deletion_nbs[0], deletion_nbs[1], deletion_nbs[2]))
        print('Syn connected: {} {} {}'.format(n_cons[0], n_cons[1], n_cons[2]))

# add weight matrix
weights_storage = {'weight_list': sess.run(weight_list),
                   'theta_list': sess.run(theta_list)}

del sess

with open('results/deep_r_results.pickle', 'wb') as f:
    pickle.dump(results, f)

with open('results/deep_r_final_weights.pickle', 'wb') as f:
    pickle.dump(weights_storage, f, protocol=pickle.HIGHEST_PROTOCOL)

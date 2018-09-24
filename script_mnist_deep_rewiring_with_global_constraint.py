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
tf.app.flags.DEFINE_float('lr_epoch_decay', 0.8, 'Learning rate decay')

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
lr = tf.Variable(FLAGS.lr, trainable=False, dtype=tf.float32)
decay_lr_op = tf.assign(lr, lr * FLAGS.lr_epoch_decay)

# Define the number of neurons per layer
sparsity_list = [FLAGS.p01, FLAGS.p02, FLAGS.p0out]
nb_non_zero_coeff_list = [n_pixels * n_1 * FLAGS.p01, n_1 * n_2 * FLAGS.p02, n_2 * n_out * FLAGS.p0out]
nb_non_zero_coeff_list = [int(n) for n in nb_non_zero_coeff_list]

# Placeholders
x = tf.placeholder(dtype, [None, n_pixels])
y = tf.placeholder(dtype, [None, n_out])


def sample_matrix_specific_reconnection_number_for_global_fixed_connectivity(theta_list, ps, upper_bound_check=False):
    with tf.name_scope('NBreconnectGenerator'):
        theta_vals = [theta.read_value() for theta in theta_list]

        # Compute size and probability of connections
        nb_possible_connections_list = [tf.cast(tf.size(th), dtype=tf.float32) * p for th, p in zip(theta_list, ps)]
        total_possible_connections = tf.reduce_sum(nb_possible_connections_list)
        max_total_connections = tf.cast(total_possible_connections, dtype=tf.int32)
        sampling_probs = [nb_possible_connections / total_possible_connections \
                          for nb_possible_connections in nb_possible_connections_list]

        def nb_connected(theta_val):
            is_con = tf.greater(theta_val, 0)
            n_connected = tf.reduce_sum(tf.cast(is_con, tf.int32))
            return n_connected

        total_connected = tf.reduce_sum([nb_connected(theta) for theta in theta_vals])

        if upper_bound_check:
            assert_upper_bound_check = tf.Assert(tf.less_equal(total_connected, max_total_connections),
                                                 data=[max_total_connections, total_connected],
                                                 name='RewiringUpperBoundCheck')
        else:
            assert_upper_bound_check = tf.Assert(True,
                                                 data=[max_total_connections, total_connected],
                                                 name='SkippedRewiringUpperBoundCheck')

        with tf.control_dependencies([assert_upper_bound_check]):
            nb_reconnect = tf.maximum(0, max_total_connections - total_connected)

            sample_split = tf.distributions.Categorical(probs=sampling_probs).sample(nb_reconnect)
            is_class_i_list = [tf.equal(sample_split, i) for i in range(len(theta_list))]
            counts = [tf.reduce_sum(tf.cast(is_class_i, dtype=tf.int32)) for is_class_i in is_class_i_list]
            return counts


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
        w_0 = rd.randn(n_in, n_out) / np.sqrt(n_in)  # initial weight values

        # Gererate the random mask
        is_con_0 = np.zeros((n_in, n_out), dtype=bool)
        ind_in = rd.choice(np.arange(n_in), size=nb_non_zero)
        ind_out = rd.choice(np.arange(n_out), size=nb_non_zero)
        is_con_0[ind_in, ind_out] = True

        # Generate random signs
        sign_0 = np.sign(rd.randn(n_in, n_out))

        # Define the tensorflow matrices
        th = tf.Variable(np.abs(w_0) * is_con_0, dtype=dtype, name='theta')
        w_sign = tf.Variable(sign_0, dtype=dtype, trainable=False, name='sign')
        is_connected = tf.greater(th, 0, name='mask')
        w = tf.where(condition=is_connected, x=w_sign * th, y=tf.zeros((n_in, n_out), dtype=dtype), name='weight')

        return w, w_sign, th, is_connected


def assert_connection_number(theta, targeted_number):
    '''
    Function to check during the tensorflow simulation if the number of connection in well defined after each simulation.
    :param theta:
    :param targeted_number:
    :return:
    '''
    th = theta.read_value()
    is_con = tf.greater(th, 0)

    nb_is_con = tf.reduce_sum(tf.cast(is_con, tf.int32))
    assert_is_con = tf.Assert(tf.equal(nb_is_con, targeted_number), data=[nb_is_con, targeted_number],
                              name='NumberOfConnectionCheck')
    return assert_is_con


def rewiring(theta, nb_reconnect, epsilon=1e-12):
    '''
    The rewiring operation to use after each iteration.
    :param theta:
    :param target_nb_connection:
    :return:
    '''

    with tf.name_scope('rewiring'):
        th = theta.read_value()
        is_con = tf.greater(th, 0)

        nb_reconnect = tf.maximum(nb_reconnect, 0)

        reconnect_candidate_coord = tf.where(tf.logical_not(is_con), name='CandidateCoord')

        n_candidates = tf.shape(reconnect_candidate_coord)[0]
        reconnect_sample_id = tf.random_shuffle(tf.range(n_candidates))[:nb_reconnect]
        reconnect_sample_coord = tf.gather(reconnect_candidate_coord, reconnect_sample_id, name='SelectedCoord')

        # Apply the rewiring
        reconnect_vals = tf.fill(dims=[nb_reconnect], value=epsilon, name='InitValues')
        reconnect_op = tf.scatter_nd_update(theta, reconnect_sample_coord, reconnect_vals, name='Reconnect')

        with tf.control_dependencies([reconnect_op]):
            return tf.no_op('Rewiring')


# Define the computational graph
with tf.name_scope('Layer1'):
    W_1, _, th_1, _ = weight_sampler_strict_number(n_pixels, n_1, nb_non_zero_coeff_list[0])
    a_1 = tf.matmul(x, W_1)
    z_1 = tf.nn.relu(a_1)

with tf.name_scope('Layer2'):
    W_2, _, th_2, _ = weight_sampler_strict_number(n_1, n_2, nb_non_zero_coeff_list[1])
    a_2 = tf.matmul(z_1, W_2)
    z_2 = tf.nn.relu(a_2)

with tf.name_scope('OutLayer'):
    w_out, _, th_out, _ = weight_sampler_strict_number(n_2, n_out, nb_non_zero_coeff_list[2])
    logits_predict = tf.matmul(z_2, w_out)

# Make list of weight for convenience
theta_list = [th_1, th_2, th_out]
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
    asserts = [assert_connection_number(th, nb) for th, nb in zip(theta_list, nb_non_zero_coeff_list)]

    with tf.control_dependencies([apply_gradients] + add_gradient_op_list + apply_l1_reg):
        number_of_rewired_connections = sample_matrix_specific_reconnection_number_for_global_fixed_connectivity(
            theta_list, [FLAGS.p01, FLAGS.p02, FLAGS.p0out])

        apply_rewiring = [rewiring(th, nb_reconnect=nb) for th, nb in zip(theta_list, number_of_rewired_connections)]
        with tf.control_dependencies(apply_rewiring):
            train_step = tf.no_op('Train')

# Some statistics for monitoring the simulation
with tf.name_scope('Stats'):
    nb_zeros = [tf.reduce_sum(tf.cast(tf.equal(w, 0), dtype)) for w in weight_list]
    sizes = [tf.cast(tf.size(w), dtype=dtype) for w in weight_list]
    layer_connectivity = [tf.cast(1, dtype=dtype) - nb_z / size for w, nb_z, size in zip(weight_list, nb_zeros, sizes)]
    global_connectivity = tf.cast(1, dtype=dtype) - tf.reduce_sum(nb_zeros) / tf.reduce_sum(sizes)

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

turnover = [0, 0, 0]
training_time = 0
acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
last_epoch = 0
for k in range(n_iter):
    if last_epoch + 1 == mnist.train.epochs_completed:
        sess.run(decay_lr_op)
        last_epoch = mnist.train.epochs_completed
        print('Learning rate decayed: {:.2g}'.format(sess.run(lr)))

    layer_connectivity_numpy = sess.run(layer_connectivity)
    global_connectivity_numpy = sess.run(global_connectivity)

    if np.mod(k, print_every) == 0:
        t0 = time.time()
        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
        testing_time = time.time() - t0

        print(
            'Epoch: {} \t time/it: {:.3g} s \t time/test: {:.3g} s  \t it: {} \t acc: {:.3g} \t loss {:.3g} \t sparsity: {:.3g} \t layer wise:'.format(
                mnist.train.epochs_completed, training_time, testing_time, k, acc, loss,
                global_connectivity_numpy) + np.array_str(
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
        print('Syn connected: {} {} {} (total: {})'.format(n_cons[0], n_cons[1], n_cons[2], n_cons[0] + n_cons[1] + n_cons[2]))

# add weight matrix
weights_storage = {'weight_list': sess.run(weight_list),
                   'theta_list': sess.run(theta_list)}

del sess

with open('results/deep_r_results.pickle', 'wb') as f:
    pickle.dump(results, f)

with open('results/deep_r_final_weights.pickle', 'wb') as f:
    pickle.dump(weights_storage, f, protocol=pickle.HIGHEST_PROTOCOL)

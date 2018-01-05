import tensorflow as tf
import numpy as np
import numpy.random as rd
from tensorflow.examples.tutorials.mnist import input_data
import os
import pickle
import time

mnist = input_data.read_data_sets("../datasets/MNIST", one_hot=True)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('n_epochs', 10, 'number of iteration (55000 is one epoch)')
tf.app.flags.DEFINE_integer('batch', 10, 'Batch size')
tf.app.flags.DEFINE_integer('print_every', 500, 'Print every')
#
tf.app.flags.DEFINE_float('p01', .005, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('p02', .005, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('p0out', .1, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('l1', 1e-5, 'L1 reg coeffcient')
tf.app.flags.DEFINE_float('gdnoise', 1e-5, 'Noise amplitude')
tf.app.flags.DEFINE_float('lr', 0.05, 'Learning rate')
#
tf.app.flags.DEFINE_string('comment', '', 'Comment to retrieve the results faster')


dtype = tf.float32

# Parameters
n_pixels = 28 * 28
n_1 = 300
n_2 = 100
n_out = 10
lr = FLAGS.lr
print_every = FLAGS.print_every
n_iter = FLAGS.n_epochs * 55000 // FLAGS.batch


def weight_sampler_with_clip(n_in, n_out, p0, l1, noise, lr):
    with tf.name_scope('SynapticSampler'):

        # Sign of the weights
        w_sign = np.sign(rd.randn(n_in,n_out))

        # Compute a rule of thumb to find a good clipping value
        assert(noise > 0)
        T = lr * noise**2 / 2
        beta = l1 / T
        p_positive = p0
        p_negative = 1 - p_positive
        th_clip = - p_negative / (p_positive * beta)

        # initil variable
        th_0 = np.where(rd.rand(n_in, n_out) < p0, rd.randn(n_in, n_out) / np.sqrt(n_in),
                        rd.rand(n_in, n_out) * th_clip)

        # Define the variables
        th = tf.Variable(th_0, dtype=tf.float32, name='theta')
        w_sign = tf.constant(w_sign, dtype=tf.float32)

        is_connected = tf.greater(th,0)

        w = tf.where(condition=is_connected, x=w_sign * th, y=tf.zeros((n_in, n_out), dtype=tf.float32))

    return w,w_sign,th,is_connected,th_clip

# Placeholders
x = tf.placeholder(dtype, [None, n_pixels])
y = tf.placeholder(dtype, [None, n_out])


# Define parameters for l2 reg on disconnected synapses

with tf.name_scope('Layer1'):
    W_1, W_sign_1, th_1, is_connected_1,th_clip_1 = weight_sampler_with_clip(n_pixels, n_1, p0=FLAGS.p01, l1=FLAGS.l1,noise=FLAGS.gdnoise,lr=lr)
    a_1 = tf.matmul(x, W_1)
    z_1 = tf.nn.relu(a_1)

with tf.name_scope('Layer2'):
    W_2, W_sign_2, th_2, is_connected_2,th_clip_2 = weight_sampler_with_clip(n_1, n_2, p0=FLAGS.p02, l1=FLAGS.l1,noise=FLAGS.gdnoise,lr=lr)
    a_2 = tf.matmul(z_1, W_2)
    z_2 = tf.nn.relu(a_2)

with tf.name_scope('OutLayer'):
    w_out, w_sign_out, th_out, is_connected_out,th_clip_out = weight_sampler_with_clip(n_2, n_out, p0=FLAGS.p0out,l1=FLAGS.l1,noise=FLAGS.gdnoise,lr=lr)
    logits_predict = tf.matmul(z_2, w_out)

theta_list = [th_1, th_2, th_out]
weight_lists = [W_1, W_2, w_out]
th_clip_list = [th_clip_1,th_clip_2,th_clip_out]

print('THETA CLIPs:')
print(th_clip_list)

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_predict))
    is_correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=dtype))

with tf.name_scope('Regularization'):
    apply_grads = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

    connected_mask = lambda th: tf.cast(tf.greater(th.read_value(), 0), tf.float32)
    l1_noisy_update = lambda th: tf.random_normal(shape=tf.shape(th), stddev=FLAGS.gdnoise,
                                                  dtype=tf.float32) - FLAGS.l1 * connected_mask(th)

    with tf.control_dependencies([apply_grads]):
        new_ths = [th + lr * l1_noisy_update(th) for th in theta_list]
        new_ths = [tf.maximum(new_th,th_clip) for new_th,th_clip in zip(new_ths,th_clip_list)]
        updates = [tf.assign(th,new_th) for th,new_th in zip(theta_list,new_ths)]
        with tf.control_dependencies(updates):
            train_step = tf.no_op('Train')

with tf.name_scope('Stats'):
    nb_zeros = [tf.reduce_sum(tf.cast(tf.equal(w, 0), dtype)) for w in weight_lists]
    sizes = [tf.cast(tf.size(w), dtype=dtype) for w in weight_lists]
    layer_connectivity = [tf.cast(1, dtype=dtype) - nb_z / size for w, nb_z, size in zip(weight_lists, nb_zeros, sizes)]
    global_connectivity = tf.cast(1, dtype=dtype) - tf.reduce_sum(nb_zeros) / tf.reduce_sum(sizes)

#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

results = {
    'loss_list': [],
    'acc_list': [],
    'global_connectivity_list': [],
    'layer_connectivity_list': [],
    'iteration_list': [],
    'epoch_list': [],
    'training_time_list': []}

training_time = 0
acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
for k in range(n_iter):
    layer_connectivity_numpy = sess.run(layer_connectivity)
    global_connectivity_numpy = sess.run(global_connectivity)
    if np.mod(k, print_every) == 0:
        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y: mnist.test.labels})

        print(
            'Epoch: {} \t iter: {} \t time/it: {:.2g} s \t acc: {:.3g} \t loss {:.2g} \t global connectivity: {:.3g} \t perlayer: '.format(
                mnist.train.epochs_completed, k, training_time, acc, loss, global_connectivity_numpy) + np.array_str(
                np.array(layer_connectivity_numpy), precision=3))

    for key, variable in zip(
            ['loss', 'acc', 'global_connectivity', 'layer_connectivity', 'iteration', 'epoch', 'training_time'],
            [loss, acc, global_connectivity_numpy, layer_connectivity_numpy, k, mnist.train.epochs_completed,
             training_time]):
        results[key + '_list'].append(variable)

    if np.mod(k, print_every) == 0:
        th_np_old = sess.run(theta_list)

    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch)
    t0 = time.time()
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    training_time = time.time() - t0

    if np.mod(k, print_every) == 0:
        th_np_new = sess.run(theta_list)
        creation_nbs = [np.sum(np.logical_and(w_new > 0, w_old <= 0)) for w_new, w_old in zip(th_np_new, th_np_old)]
        deletion_nbs = [np.sum(np.logical_and(w_new <= 0, w_old > 0)) for w_new, w_old in zip(th_np_new, th_np_old)]

        print('Syn created: {} {} {}'.format(creation_nbs[0], creation_nbs[1], creation_nbs[2]))
        print('Syn deleted: {} {} {}'.format(deletion_nbs[0], deletion_nbs[1], deletion_nbs[2]))


# add weight matrix
weights_storage = {}
weights_storage['th_1'] = sess.run(th_1)
weights_storage['w_1'] = sess.run(W_1)
weights_storage['th_2'] = sess.run(th_2)
weights_storage['w_2'] = sess.run(W_2)
weights_storage['th_out'] = sess.run(th_out)
weights_storage['w_out'] = sess.run(w_out)

del sess


with open('results/soft_deep_r_results.pickle', 'wb') as f:
    pickle.dump(results, f)

with open('results/soft_deep_r_final_weights.pickle', 'wb') as f:
    pickle.dump(weights_storage, f, protocol=pickle.HIGHEST_PROTOCOL)

import os
import numpy as np
import tensorflow as tf
import tensorbayes as tb
from config import args
from tensorbayes.layers import *
from models import *
from data import CelebA
from utils import plot_results, load_model, save_model

tf.reset_default_graph()
x_real = placeholder((None, args.i_size, args.i_size, 3), name='x')
z = placeholder((None, args.e_size), name='z')
k = placeholder(None, name='k')
lr = placeholder(None, name='lr')
gamma = placeholder(None, name='g')

# Generate
x_fake = generator(z)

# Discriminate
d_real = decoder(encoder(x_real))
d_fake = decoder(encoder(x_fake, reuse=True), reuse=True)

# Loss
d_real_loss = tf.reduce_mean(tf.abs(x_real - d_real))
d_fake_loss = tf.reduce_mean(tf.abs(x_fake - d_fake))
d_loss = d_real_loss - k * d_fake_loss
g_loss = d_fake_loss
m_global = d_real_loss + tf.abs(gamma * d_real_loss - d_fake_loss)

# Optimizer
d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc*')
g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen*')
print [v.name for v in d_var]
print [v.name for v in g_var]
d_train = tf.train.AdamOptimizer(lr, 0.5).minimize(d_loss, var_list=d_var)
g_train = tf.train.AdamOptimizer(lr, 0.5).minimize(g_loss, var_list=g_var)

# Logger
base_dir = 'results/gamma={:.1f}_run={:d}'.format(args.gamma, args.run)
writer = tb.FileWriter(os.path.join(base_dir, 'log.csv'), args=args, overwrite=args.run >= 999)
writer.add_var('d_real', '{:8.4f}', d_real_loss)
writer.add_var('d_fake', '{:8.4f}', d_fake_loss)
writer.add_var('k', '{:8.4f}', k * 1)
writer.add_var('M', '{:8.4f}', m_global)
writer.add_var('lr', '{:8.6f}', lr * 1)
writer.add_var('iter', '{:>8d}')
writer.initialize()

sess = tf.Session()
load_model(sess)
f_gen = tb.function(sess, [z], x_fake)
f_rec = tb.function(sess, [x_real], d_real)
celeba = CelebA(args.data)

# Alternatively try grouping d_train/g_train together
all_tensors = [d_train, g_train, d_real_loss, d_fake_loss]
# d_tensors = [d_train, d_real_loss]
# g_tensors = [g_train, d_fake_loss]
for i in xrange(args.max_iter):
    x = celeba.next_batch(args.bs)
    z = np.random.uniform(-1, 1, (args.bs, args.e_size))

    feed_dict = {'x:0': x, 'z:0': z, 'k:0': args.k, 'lr:0': args.lr, 'g:0': args.gamma}
    _, _, d_real_loss, d_fake_loss = sess.run(all_tensors, feed_dict)
    # _, d_real_loss = sess.run(d_tensors, feed_dict)
    # _, d_fake_loss = sess.run(g_tensors, feed_dict)
    args.k = np.clip(args.k + args.lambd * (args.gamma * d_real_loss - d_fake_loss), 0., 1.)

    msg = '{:8.4f}/{:8.4f}/{:8.6f}'.format(d_real_loss, d_fake_loss, args.k)
    tb.utils.progbar(i, args.log_step, message=msg, bar_length=20)

    if (i + 1) % args.log_step == 0:
        x = celeba.next_batch(args.bs)
        z = np.random.uniform(-1, 1, (args.bs, args.e_size))
        feed_dict = {'x:0': x, 'z:0': z, 'k:0': args.k, 'lr:0': args.lr, 'g:0': args.gamma}
        val = sess.run(writer.tensors, feed_dict)
        writer.write(tensor_values=val, values=[i + 1])

    if (i + 1) % args.plot_step == 0:
        plot_results(celeba, f_rec, f_gen, i, base_dir)

    if (i + 1) % args.lr_step == 0:
        args.lr *= 0.5

    if (i + 1) % args.save_step == 0:
        save_model()

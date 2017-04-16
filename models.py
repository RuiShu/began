import tensorflow as tf
from config import args
from tensorbayes.layers import *

def encoder(x, scope='disc_encoder', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        x = conv2d(x, args.f_size, 3, 1, activation=tf.nn.elu)
        # Encoder
        i = 1
        while True:
            x = conv2d(x, i * args.f_size, 3, 1, activation=tf.nn.elu)
            x = conv2d(x, i * args.f_size, 3, 1, activation=tf.nn.elu)
            if x._shape_as_list()[1] <= 8: break
            # x = conv2d(x, (i + 1) * args.f_size, 3, 1)
            x = avg_pool(x, 2, 2)
            # x = conv2d(x, (i + 1) * args.f_size, 3, 2, activation=tf.nn.elu)
            i += 1

        x = dense(x, args.e_size)
    return x

def decoder(x, scope='disc_decoder', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        x = tf.reshape(dense(x, 8 * 8 * args.f_size), [-1, 8, 8, args.f_size])

        while True:
            x = conv2d(x, args.f_size, 3, 1, activation=tf.nn.elu)
            x = conv2d(x, args.f_size, 3, 1, activation=tf.nn.elu)
            if x._shape_as_list()[1] >= args.i_size: break
            x = upsample(x, 2)

        x = conv2d(x, 3, 3, 1)
    return x

def generator(z, scope='generator', reuse=None):
    return decoder(z, scope, reuse)

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import args

def plot_results(celeba, f_rec, f_gen, step, base_dir):
    z = np.random.uniform(-1, 1, (16, args.e_size))
    x_fake = f_gen(z)
    x_real = celeba.next_batch(16)
    x_fake_rec = f_rec(x_fake)
    x_real_rec = f_rec(x_real)

    x_fake = x_fake.reshape(16 * 64, 64, 3)
    x_real = x_real.reshape(16 * 64, 64, 3)
    x_fake_rec = x_fake_rec.reshape(16 * 64, 64, 3)
    x_real_rec = x_real_rec.reshape(16 * 64, 64, 3)
    x = celeba.denorm(np.concatenate((x_fake, x_fake_rec, x_real, x_real_rec), axis=1))

    plt.figure(figsize=(10, 10))
    plt.imshow(x, interpolation='None')
    plt.axis('off')
    fpath = os.path.join(base_dir, 'output={:08d}.png'.format(step + 1))
    plt.savefig(fpath, dpi=300, bbox='tight')

def load_model(sess):
    sess.run(tf.global_variables_initializer())

def save_model():
    pass


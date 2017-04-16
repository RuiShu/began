import os
import numpy as np
from scipy.io import loadmat

class CelebA(object):
    def __init__(self, data_dir):
        self._load(data_dir)

    def _load(self, data_dir):
        print 'Preparing file'
        x = loadmat(data_dir)['images']
        x = x.astype('float32')
        x /= 255.
        x -= 0.5
        self.x = x

    def norm(self, x):
        return x - 0.5

    def denorm(self, x):
        return np.clip(x + 0.5, 0, 1)

    def next_batch(self, bs):
        idx = np.random.choice(len(self.x), bs, replace=False)
        return self.x[idx]

if __name__ == '__main__':
    celeba = CelebA()
    print celeba.x[:10000].min(), celeba.x[:10000].max()

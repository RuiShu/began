import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run",       type=int,   default=999,    help="Experiment run ID")
parser.add_argument("--i-size",    type=int,   default=64,     help="Image size")
parser.add_argument("--f-size",    type=int,   default=64,     help="Filter size")
parser.add_argument("--e-size",    type=int,   default=128,    help="Embedding size")
parser.add_argument("--gamma",     type=float, default=0.5,    help="Gamma equilibrium ratio")
parser.add_argument("--bs",        type=int,   default=16,     help="Minibatch size.")
parser.add_argument("--lr",        type=float, default=1e-4,   help="Learning rate.")
parser.add_argument("--lambd",     type=float, default=1e-3,   help="Equilibrium control.")
parser.add_argument("--k",         type=float, default=0.,     help="Initial coefficient.")
parser.add_argument("--max-iter",  type=int,   default=500000, help="Number of iterations.")
parser.add_argument("--lr-step",   type=int,   default=100000, help="Decay learning rate after # steps.")
parser.add_argument("--log-step",  type=int,   default=100,    help="Log after # steps.")
parser.add_argument("--plot-step", type=int,   default=2000,   help="Plot after # steps.")
parser.add_argument("--save-step", type=int,   default=5000,   help="Plot after # steps.")
parser.add_argument("--data-dir",  type=str,   default='/local-scratch/rshu15/celeba/celeba_64_zoom.mat', help="Data directory.")
args = parser.parse_args()
print args

def get_config():
    return args

from PIL import Image
from scipy.io import savemat
import numpy as np
from glob import glob
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--source-dir", type=str,
                    default='/local-scratch/rshu15/celeba/img_align_celeba/',
                    help="Source directory for celeba")
parser.add_argument("--dest-file",   type=str,
                    default='/local-scratch/rshu15/celeba/celeba_64_zoom.mat',
                    help="Destination file for 64 x 64 celeba .mat file")
args = parser.parse_args()

files = np.sort(glob(os.path.join(args.source_dir, '*')))
assert len(files) != 0, "Unable to find any files in source directory"
img_buf = np.zeros((len(files), 64, 64, 3), dtype='uint8')

for i, f in enumerate(files):
    img = Image.open(f).crop((25, 50, 25 + 128, 50 + 128)).resize((64, 64), Image.ANTIALIAS)
    img_buf[i] = np.array(img)
    print '{:d}/{:d}'.format(i + 1, len(files))

savemat(args.dest_file, {'images': img_buf})

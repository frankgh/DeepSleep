import os
import math
import glob
import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":

    ap = ArgumentParser(description='Generate the folder structure for training')
    ap.add_argument('-i', dest='data_dir', metavar='data_dir', help='path for the npz patient data', required=True)
    ap.add_argument('-o', dest='output_dir', metavar='output_dir', help='path for the output directory', required=True)
    ap.add_argument('--folds', dest='k_folds', metavar='k_folds', help='number of folds for training', type=int,
                    default=10)
    args = ap.parse_args()

    data_files = np.array(glob.glob(os.path.join(args.data_dir, '*.np[yz]')))
    perm = np.random.permutation(len(data_files))
    fold_size = int(math.ceil(len(data_files) / args.k_folds))

    for k in range(args.k_folds):
        i = int(k * fold_size)
        test_files = data_files[perm[i:i + fold_size]]
        dir_name = os.path.join(args.output_dir,
                                ''.join([os.path.splitext(os.path.basename(name))[0] for name in test_files]))

        if not os.path.exists(dir_name):
            print 'Creating', dir_name
            os.makedirs(dir_name)

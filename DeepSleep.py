import time
import numpy
import random
from datetime import timedelta
from argparse import ArgumentParser

from DeepSleepClassifier import DeepSleepClassifier

# For reproducibility
random.seed(4236)
numpy.random.seed(4236)

if __name__ == "__main__":
    ap = ArgumentParser(description='Train the deep sleep neural network')
    ap.add_argument('-i', dest='data_dir', metavar='data_dir', help='path for the npz patient data', required=True)
    ap.add_argument('-o', dest='output_dir', metavar='output_dir', help='path for the output directory', required=True)
    ap.add_argument('--folds', dest='k_folds', metavar='k_folds', help='number of folds for training')
    ap.add_argument('--bs', dest='batch_size', metavar='batch_size', help='batch size')
    ap.add_argument('--epochs', dest='epochs', metavar='epochs', help='number of epochs')
    ap.add_argument('--lr', dest='lr', metavar='learning rate', help='learning rate')
    ap.add_argument('--decay', dest='decay', metavar='decay', help='decay of learning rate')
    ap.add_argument('--mtn', dest='m', metavar='momentum', help='momentum')
    ap.add_argument('--ridge', dest='ridge', metavar='ridge', help='ridge term for l2 regularization')
    ap.add_argument('--patience', dest='patience', metavar='patience', help='patience for early stopping')
    ap.add_argument('--init', dest='kernel_initializer', metavar='kernel_initializer', help='kernel initializer')
    ap.add_argument('--verbose', dest='verbose', metavar='verbose', help='verbosity level')
    args = ap.parse_args()

    print 'Setting up'

    start = time.time()
    classifier = DeepSleepClassifier(args.data_dir, args.output_dir,
                                     k_folds=args.k_folds,
                                     batch_size=args.batch_size,
                                     epochs=args.epochs,
                                     lr=args.lr,
                                     decay=args.decay,
                                     m=args.m,
                                     ridge=args.ridge,
                                     patience=args.patience,
                                     kernel_initializer=args.kernel_initializer,
                                     verbose=args.verbose)
    model, _ = classifier.train_model()
    classifier.test_model(model)
    elapsed = time.time() - start

    print 'Training completed in', str(timedelta(seconds=elapsed))

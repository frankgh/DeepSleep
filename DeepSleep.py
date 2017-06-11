import time
from argparse import ArgumentParser
from datetime import timedelta

from DeepSleepClassifier import DeepSleepClassifier


# For reproducibility
# random.seed(8512)
# numpy.random.seed(8512)


def get_kwargs(args):
    _kwargs = dict()
    if args.batch_size is not None:
        _kwargs['batch_size'] = args.batch_size
    if args.epochs is not None:
        _kwargs['epochs'] = args.epochs
    if args.lr is not None:
        _kwargs['lr'] = args.lr
    if args.decay is not None:
        _kwargs['decay'] = args.decay
    if args.m is not None:
        _kwargs['m'] = args.m
    if args.ridge is not None:
        _kwargs['ridge'] = args.ridge
    if args.patience is not None:
        _kwargs['patience'] = args.patience
    if args.kernel_initializer is not None:
        _kwargs['kernel_initializer'] = args.kernel_initializer
    if args.verbose is not None:
        _kwargs['verbose'] = args.verbose
    if args.conv_layers is not None:
        _kwargs['convolutional_layers'] = args.conv_layers
    if args.iter is not None:
        _kwargs['iterations'] = args.iter
    if args.strides is not None:
        _kwargs['strides'] = args.strides
    if args.istrides is not None:
        _kwargs['initial_strides'] = args.istrides
    if args.ifilters is not None:
        _kwargs['initial_filters'] = args.ifilters
    if args.iks is not None:
        _kwargs['initial_kernel_size'] = args.iks
    return _kwargs


if __name__ == "__main__":
    ap = ArgumentParser(description='Train the Deep Sleep neural network')
    ap.add_argument('-i', dest='data_dir', metavar='data_dir', help='path for the npz patient data', required=True)
    ap.add_argument('-o', dest='output_dir', metavar='output_dir', help='path for the output directory', required=True)
    ap.add_argument('--bs', dest='batch_size', metavar='batch_size', help='batch size', type=int)
    ap.add_argument('--epochs', dest='epochs', metavar='epochs', help='number of epochs', type=int)
    ap.add_argument('--lr', dest='lr', metavar='learning rate', help='learning rate', type=float)
    ap.add_argument('--decay', dest='decay', metavar='decay', help='decay of learning rate', type=float)
    ap.add_argument('--mtn', dest='m', metavar='momentum', help='momentum', type=float)
    ap.add_argument('--ridge', dest='ridge', metavar='ridge', help='ridge term for l2 regularization', type=float)
    ap.add_argument('--patience', dest='patience', metavar='patience', help='patience for early stopping', type=int)
    ap.add_argument('--init', dest='kernel_initializer', metavar='kernel_initializer', help='kernel initializer')
    ap.add_argument('--verbose', dest='verbose', metavar='verbose', help='verbosity level', type=int)
    ap.add_argument('--conv_layers', dest='conv_layers', metavar='convolutional layers',
                    help='number of convolutional layers', type=int)
    ap.add_argument('--iter', dest='iter', metavar='iterations', help='number of iterations', type=int)
    ap.add_argument('--strides', dest='strides', metavar='strides', help='strides for convolutional layers', type=int)
    ap.add_argument('--istrides', dest='istrides', metavar='initial strides',
                    help='strides for the first convolutional layer', type=int)
    ap.add_argument('--ifilters', dest='ifilters', metavar='initial filters',
                    help='filters for the first convolutional layer', type=int)
    ap.add_argument('--iks', dest='iks', metavar='initial kernel size',
                    help='kernel size for the first convolutional layer', type=int)
    args = ap.parse_args()
    kwargs = get_kwargs(args)

    print 'Setting up with kwargs:', kwargs

    start = time.time()
    classifier = DeepSleepClassifier(args.data_dir, args.output_dir, **kwargs)
    model, _ = classifier.train_model()
    classifier.test_model(model)
    elapsed = time.time() - start

    print 'Classifier config:', classifier.get_config()
    print 'Training completed in', str(timedelta(seconds=elapsed))

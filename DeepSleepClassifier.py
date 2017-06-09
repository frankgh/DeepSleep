import os
import glob
import math
import numpy as np
import itertools
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt

from sklearn.utils import compute_class_weight

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Recurrent
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.initializers import Constant

matplotlib.use('Agg')


def next_batch(data, size):
    for item in itertools.cycle(data):
        perm = np.random.permutation(item['Y'].shape[0])
        for i in np.arange(0, item['Y'].shape[0], size):
            yield (item['X'][perm[i:i + size]], item['Y'][perm[i:i + size]])


def unfold(data, verbose=0):
    x, y = np.array(data[0]['X']), np.array(data[0]['Y'])
    if verbose > 0:
        print 'Unfolding: '
        print ' -', data[0]['name']
    for item in data[1:]:
        x = np.concatenate((x, item['X']))
        y = np.concatenate((y, item['Y']))
        if verbose > 0:
            print ' -', item['name']
    return x, y


def count_samples(data):
    return np.sum([item['Y'].shape[0] for item in data])


def count_steps(data, batch_size):
    return int(np.sum([item['Y'].shape[0] / batch_size for item in data]))


def calculate_weights(data):
    y = np.array(data[0]['Y'])
    for item in data[1:]:
        y = np.concatenate((y, item['Y']))
    y_1 = np.argmax(y, axis=1)
    return compute_class_weight('balanced', np.arange(5), y_1)


def plot_confusion_matrix(output_dir, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'conf_mat.png'), dpi=100)
    plt.clf()
    plt.cla()
    plt.close()


def plot_roc_curve(output_dir, n_classes, y_true, y_pred):
    """
    Compute ROC curve and ROC area for each class
    :param output_dir: where to save the png image file 
    :param n_classes: number of classes
    :param y_true: the true values
    :param y_pred: predicted values
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_score.png'), dpi=100)
    plt.clf()
    plt.cla()
    plt.close()

    roc_score = metrics.roc_auc_score(y_true, y_pred)
    print "ROC AUC Score: ", roc_score


def plot_accuracy(output_dir, acc, val_acc, splits):
    """
    Summarize history for accuracy
    :param output_dir: the output directory for the plot png file 
    :param acc: training accuracy list
    :param val_acc: validation accuracy list
    """
    plt.plot(acc, linewidth=2)
    plt.plot(val_acc, linestyle='dotted', linewidth=2)

    total = 0
    for n in splits:
        total += n
        plt.axvline(x=total, linestyle='dotted', linewidth=1, color='r')

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='bottom right')
    plt.savefig(os.path.join(output_dir, 'accuracy.png'), dpi=100)
    plt.clf()
    plt.cla()
    plt.close()


def plot_loss(output_dir, loss, val_loss, splits):
    """
    Summarize history for loss
    :param output_dir: the output directory for the plot png file
    :param loss: training loss history
    :param val_loss: validation loss history
    """
    plt.plot(loss, linewidth=2)
    plt.plot(val_loss, linestyle='dotted', linewidth=2)

    total = 0
    for n in splits:
        total += n
        plt.axvline(x=total, linestyle='dotted', linewidth=1, color='r')

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=100)
    plt.clf()
    plt.cla()
    plt.close()


class DeepSleepClassifier(object):
    def __init__(self, data_dir,
                 output_dir,
                 input_weights_filepath,
                 k_folds=9,
                 batch_size=192,
                 epochs=100,
                 lr=1e-5,
                 decay=0.9,
                 m=0.5,
                 ridge=2e-4,
                 patience=10,
                 kernel_initializer='he_normal',
                 convolutional_layers=3,
                 verbose=2,
                 iterations=1,
                 filters=25,
                 kernel_size=50):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.input_weights_filepath = input_weights_filepath
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
        self.m = m
        self.ridge = ridge
        self.patience = patience
        self.kernel_initializer = kernel_initializer
        self.verbose = verbose
        self.filters = filters
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.convolutional_layers = convolutional_layers

        self.data = self.load_data()
        self.train_set, self.test_set = self.split_data()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """
        Load all npz files from the given path
        :param self: self
        """
        return np.array([np.load(np_name) for np_name in glob.glob(os.path.join(self.data_dir, '*.np[yz]'))])

    def split_data(self, split=0.1):
        """
        Split permutated data into train and test set split by the split value
        :param self: 
        :param split: the split amount
        :return: the training and test sets
        """
        i = int(len(self.data) * split)
        perm = np.random.permutation(len(self.data))  # permute data

        if self.verbose > 0:
            print 'Test elements:'
            for item in self.data[perm[0:i]]:
                print ' -', item['name']

        return self.data[perm[i:]], self.data[perm[0:i]]  # return training, test sets

    def build_model(self):
        optimizer = RMSprop(lr=self.lr, decay=self.decay)
        bias_init = Constant(value=0.1)
        model = Sequential()

        model.add(Conv1D(64, 500, padding='valid', input_shape=(15000, 3), name='conv1d_1'))
        model.add(BatchNormalization(name='batch_normalization_1'))
        model.add(LeakyReLU(alpha=0.3, name='leaky_re_lu_1'))

        model.add(Conv1D(self.filters, self.kernel_size, padding='valid', name='conv1d_2'))
        model.add(BatchNormalization(name='batch_normalization_2'))
        model.add(LeakyReLU(alpha=0.3, name='leaky_re_lu_2'))

        model.add(Conv1D(self.filters, self.kernel_size, padding='valid', name='conv1d_3'))
        model.add(BatchNormalization(name='batch_normalization_3'))
        model.add(LeakyReLU(alpha=0.3, name='leaky_re_lu_3'))

        model.add(Conv1D(self.filters, self.kernel_size, padding='valid', name='conv1d_4'))
        model.add(BatchNormalization(name='batch_normalization_4'))
        model.add(LeakyReLU(alpha=0.3, name='leaky_re_lu_4'))

        model.add(Conv1D(self.filters, self.kernel_size, padding='valid', name='conv1d_5'))
        model.add(BatchNormalization(name='batch_normalization_5'))
        model.add(LeakyReLU(alpha=0.3, name='leaky_re_lu_5'))

        model.add(Conv1D(self.filters, self.kernel_size, padding='valid', name='new_conv1d_6'))
        model.add(BatchNormalization(name='new_batch_normalization_6'))
        model.add(LeakyReLU(alpha=0.3, name='new_leaky_re_lu_6'))

        model.add(Conv1D(self.filters, self.kernel_size, padding='valid', name='new_conv1d_7'))
        model.add(BatchNormalization(name='new_batch_normalization_7'))
        model.add(LeakyReLU(alpha=0.3, name='new_leaky_re_lu_7'))

        model.add(MaxPooling1D(name='max_pooling1d_1'))
        model.add(Flatten(name='flatten_1'))

        model.add(Dense(512, kernel_initializer=self.kernel_initializer, bias_initializer=bias_init,
                        kernel_regularizer=l2(self.ridge), name='dense_1'))
        model.add(BatchNormalization(name='batch_normalization_6'))
        model.add(LeakyReLU(alpha=0.3, name='leaky_re_lu_6'))

        model.add(Dense(128, kernel_initializer=self.kernel_initializer, bias_initializer=bias_init,
                        kernel_regularizer=l2(self.ridge), name='dense_2'))
        model.add(BatchNormalization(name='batch_normalization_7'))
        model.add(LeakyReLU(alpha=0.3, name='leaky_re_lu_7'))

        model.add(Dense(5, kernel_initializer=self.kernel_initializer, bias_initializer=bias_init,
                        kernel_regularizer=l2(self.ridge), activation='softmax', name='dense_3'))

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self):
        model = self.build_model()

        if self.verbose > 0:
            print 'Loading weights from ' + self.input_weights_filepath

        model.load_weights(self.input_weights_filepath, by_name=True)
        model.summary()
        fold_size = int(math.ceil(len(self.train_set) / self.k_folds))
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=self.verbose,
                                      mode='auto')
        class_weight = calculate_weights(self.train_set)
        acc, val_acc, loss, val_loss, splits = [], [], [], [], []

        for k in range(self.iterations * self.k_folds):
            i = int(k * fold_size) % self.k_folds
            val = self.train_set[i:i + fold_size]
            train = np.concatenate((self.train_set[:i], self.train_set[i + fold_size:]))
            steps_per_epoch = count_steps(train, self.batch_size)
            if self.verbose > 0:
                print 'Fold:', (k + 1), 'Samples:', count_samples(
                    train), 'Epochs:', self.epochs, 'Steps:', steps_per_epoch

            name = 'f' + str(k + 1) + '-e' + str(self.epochs) + '-lr' + str(self.lr) + '-dcy' + str(
                self.decay) + '-m' + str(self.m) + '-reg' + str(self.ridge)
            filepath = os.path.join(self.output_dir, 'DS_' + name + '_{epoch:03d}-{val_acc:.2f}.h5')
            checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=self.verbose,
                                           save_best_only=True)

            history = model.fit_generator(next_batch(train, self.batch_size), steps_per_epoch,
                                          epochs=self.epochs,
                                          verbose=self.verbose,
                                          class_weight=class_weight,
                                          validation_data=unfold(val, self.verbose),
                                          callbacks=[checkpointer, early_stopper])

            acc.extend(history.history['acc'])
            val_acc.extend(history.history['val_acc'])
            loss.extend(history.history['loss'])
            val_loss.extend(history.history['val_loss'])
            splits.append(len(history.history['acc']))

        if self.verbose > 0:
            print(history.history.keys())
        model.save(os.path.join(self.output_dir, 'model.h5'))
        plot_accuracy(self.output_dir, acc, val_acc, splits)
        plot_loss(self.output_dir, loss, val_loss, splits)
        return model, history

    def test_model(self, model):
        # Test model on test set
        test_x, y_true = unfold(self.test_set)
        loss_and_metrics = model.evaluate(test_x, y_true, batch_size=self.batch_size, verbose=self.verbose)
        y_pred = model.predict(test_x, batch_size=self.batch_size, verbose=self.verbose)

        y_true_class = np.argmax(y_true, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        conf_mat = metrics.confusion_matrix(y_true_class, y_pred_class)

        if self.verbose > 0:
            print "Loss: {} Accuracy: {}%".format(loss_and_metrics[0], loss_and_metrics[1] * 100)
        plot_confusion_matrix(self.output_dir, conf_mat, classes=['S1', 'S2', 'S3', 'A', 'R'])
        plot_roc_curve(self.output_dir, 5, y_true, y_pred)

    def get_config(self):
        config = {
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'input_weights_filepath': self.input_weights_filepath,
            'k_folds': self.k_folds,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'decay': self.decay,
            'm': self.m,
            'ridge': self.ridge,
            'patience': self.patience,
            'kernel_initializer': self.kernel_initializer,
            'verbose': self.verbose,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'iterations': self.iterations,
            'convolutional_layers': self.convolutional_layers
        }
        return dict(list(config.items()))

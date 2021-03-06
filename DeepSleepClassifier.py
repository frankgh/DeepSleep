import glob
import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from sklearn.utils import compute_class_weight


def next_batch(X, y, size):
    while 1:
        perm = np.random.permutation(X.shape[0])
        for i in np.arange(0, X.shape[0], size):
            yield (X[perm[i:i + size]], y[perm[i:i + size]])


# def next_batch(data, size, verbose=0):
#     for item in itertools.cycle(data):
#         if verbose > 2:
#             print 'Training on -', item['name']
#         perm = np.random.permutation(item['Y'].shape[0])
#         for i in np.arange(0, item['Y'].shape[0], size):
#             yield (item['X'][perm[i:i + size]], item['Y'][perm[i:i + size]])


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


class DeepSleepClassifier(object):
    def __init__(self, data_dir,
                 output_dir,
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
                 strides=5,
                 kernel_size=50,
                 initial_filters=128,
                 initial_strides=50,
                 initial_kernel_size=500,
                 split=0.17,
                 padding='same'):
        self.data_dir = data_dir
        self.output_dir = output_dir
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
        self.strides = strides
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.convolutional_layers = convolutional_layers
        self.initial_filters = initial_filters
        self.initial_strides = initial_strides
        self.initial_kernel_size = initial_kernel_size
        self.padding = padding

        self.load_data()
        self.train_set, self.val_set = self.split_data(split=split)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """
        Load all npz files from the given path
        :param self: self
        """
        all_files = glob.glob(os.path.join(self.data_dir, '*.np[yz]'))
        test_files = ['p' + s + '.npz' for s in os.path.basename(os.path.normpath(self.output_dir)).split('p') if
                      s.isdigit()]
        self.data = np.array([np.load(f) for f in all_files if not f.endswith(tuple(test_files))])
        self.test_data = np.array([np.load(f) for f in all_files if f.endswith(tuple(test_files))])

    def split_data(self, split=0.17):
        """
        Split permutated data into train and test set split by the split value
        :param self: 
        :param split: the split amount
        :return: the training and test sets
        """
        i = int(len(self.data) * split)
        perm = np.random.permutation(len(self.data))  # permute data

        if self.verbose > 0:
            print 'Validation elements:'
            for item in self.data[perm[0:i]]:
                print ' -', item['name']

        return self.data[perm[i:]], self.data[perm[0:i]]  # return training, test sets

    def build_model(self):
        optimizer = 'sgd'
        model = Sequential()
        model.add(
            Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                   input_shape=(15000, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        for i in range(5):
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

        for i in range(3):
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

        for i in range(3):
            model.add(
                Conv1D(50, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

        for i in range(3):
            model.add(Conv1D(100, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))

        model.add(MaxPooling1D(pool_size=10, strides=10, padding='valid'))
        model.add(Conv1D(100, 100, strides=1, padding='same', kernel_initializer=self.kernel_initializer))
        model.add(MaxPooling1D(pool_size=10, strides=10, padding='valid'))

        model.add(Conv1D(4, 5, strides=1, padding='valid', kernel_initializer=self.kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(100, kernel_initializer=self.kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(100, kernel_initializer=self.kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(5, kernel_initializer=self.kernel_initializer, activation='softmax'))

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self):
        model = self.build_model()

        if self.verbose > 0:
            model.summary()
            print 'Training set:'

        train_x, train_y = unfold(self.train_set, self.verbose)
        class_weight = calculate_weights(self.train_set)
        steps_per_epoch = int(np.math.ceil(1.0 * len(train_y) / self.batch_size))

        if self.verbose > 0:
            print 'Samples:', len(train_y), 'Epochs:', self.epochs, 'Steps:', steps_per_epoch

        name = 'DS_e{0:d}-lr{1:g}-dcy{2:g}-m{3:g}-reg{4:g}'.format(self.epochs, self.lr, self.decay, self.m, self.ridge)
        file_path = os.path.join(self.output_dir, name + '_{epoch:03d}-{val_acc:.2f}.h5')
        model_check = ModelCheckpoint(filepath=file_path, monitor='val_loss', verbose=self.verbose, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, min_lr=1e-6)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=self.verbose,
                                      mode='auto')

        history = model.fit_generator(next_batch(train_x, train_y, self.batch_size), steps_per_epoch,
                                      epochs=self.epochs,
                                      verbose=self.verbose,
                                      class_weight=class_weight,
                                      validation_data=unfold(self.val_set, self.verbose),
                                      callbacks=[model_check, early_stopper, reduce_lr])

        if self.verbose > 1:
            print(history.history.keys())

        model.save(os.path.join(self.output_dir, 'model.h5'))
        np.savez(os.path.join(self.output_dir, 'history.npz'), acc=history.history['acc'],
                 val_acc=history.history['val_acc'], loss=history.history['loss'], val_loss=history.history['val_loss'])

        return model, history

    def test_model(self, model):
        # Test model on test set
        if self.verbose > 0:
            print 'Testing model'
        test_x, y_true = unfold(self.test_data, self.verbose)
        loss_and_metrics = model.evaluate(test_x, y_true, batch_size=self.batch_size, verbose=self.verbose)
        y_pred = model.predict(test_x, batch_size=self.batch_size, verbose=self.verbose)

        # y_true_class = np.argmax(y_true, axis=1)
        # y_pred_class = np.argmax(y_pred, axis=1)
        # conf_mat = metrics.confusion_matrix(y_true_class, y_pred_class)

        if self.verbose > 0:
            print "Loss: {} Accuracy: {}%".format(loss_and_metrics[0], loss_and_metrics[1] * 100)

        np.savez(os.path.join(self.output_dir, 'test_results.npz'), loss_and_metrics=loss_and_metrics,
                 y_true=y_true, y_pred=y_pred)

    def get_config(self):
        config = {
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
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
            'strides': self.strides,
            'iterations': self.iterations,
            'convolutional_layers': self.convolutional_layers
        }
        return dict(list(config.items()))

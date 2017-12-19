import glob
import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD
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
        self.data = None  # Initialized in load_data
        self.test_data = None  # Initialized in load_data
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

    def build_model(self, layers=-1):
        optimizer = SGD(lr=self.lr, decay=self.decay, nesterov=False)
        model = Sequential()
        model.add(
            Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer, name='conv1d_1',
                   input_shape=(15000, 3)))
        model.add(BatchNormalization(name='batch_normalization_1'))
        model.add(Activation('relu', name='activation_1'))

        if layers == -1 or layers >= 2:
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_2'))
            model.add(BatchNormalization(name='batch_normalization_2'))
            model.add(Activation('relu', name='activation_2'))

        if layers == -1 or layers >= 3:
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_3'))
            model.add(BatchNormalization(name='batch_normalization_3'))
            model.add(Activation('relu', name='activation_3'))

        if layers == -1 or layers >= 4:
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_4'))
            model.add(BatchNormalization(name='batch_normalization_4'))
            model.add(Activation('relu', name='activation_4'))

        if layers == -1 or layers >= 5:
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_5'))
            model.add(BatchNormalization(name='batch_normalization_5'))
            model.add(Activation('relu', name='activation_5'))

        if layers == -1 or layers >= 6:
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_6'))
            model.add(BatchNormalization(name='batch_normalization_6'))
            model.add(Activation('relu', name='activation_6'))

        if layers == -1 or layers >= 7:
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='max_pooling1d_1'))

        if layers == -1 or layers >= 8:
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_7'))
            model.add(BatchNormalization(name='batch_normalization_7'))
            model.add(Activation('relu', name='activation_7'))

        if layers == -1 or layers >= 9:
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_8'))
            model.add(BatchNormalization(name='batch_normalization_8'))
            model.add(Activation('relu', name='activation_8'))

        if layers == -1 or layers >= 10:
            model.add(
                Conv1D(25, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_9'))
            model.add(BatchNormalization(name='batch_normalization_9'))
            model.add(Activation('relu', name='activation_9'))

        if layers == -1 or layers >= 11:
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='max_pooling1d_2'))

        if layers == -1 or layers >= 12:
            model.add(
                Conv1D(50, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_10'))
            model.add(BatchNormalization(name='batch_normalization_10'))
            model.add(Activation('relu', name='activation_10'))

        if layers == -1 or layers >= 13:
            model.add(
                Conv1D(50, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_11'))
            model.add(BatchNormalization(name='batch_normalization_11'))
            model.add(Activation('relu', name='activation_11'))

        if layers == -1 or layers >= 14:
            model.add(
                Conv1D(50, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_12'))
            model.add(BatchNormalization(name='batch_normalization_12'))
            model.add(Activation('relu', name='activation_12'))

        if layers == -1 or layers >= 15:
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='max_pooling1d_3'))

        if layers == -1 or layers >= 16:
            model.add(
                Conv1D(100, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_13'))
            model.add(BatchNormalization(name='batch_normalization_13'))
            model.add(Activation('relu', name='activation_13'))

        if layers == -1 or layers >= 17:
            model.add(
                Conv1D(100, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_14'))
            model.add(BatchNormalization(name='batch_normalization_14'))
            model.add(Activation('relu', name='activation_14'))

        if layers == -1 or layers >= 18:
            model.add(
                Conv1D(100, 100, strides=1, padding='valid', kernel_initializer=self.kernel_initializer,
                       name='conv1d_15'))
            model.add(BatchNormalization(name='batch_normalization_15'))
            model.add(Activation('relu', name='activation_15'))

        if layers == -1 or layers >= 19:
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid', name='max_pooling1d_4'))

        if layers == -1 or layers >= 20:
            model.add(MaxPooling1D(pool_size=10, strides=10, padding='valid', name='max_pooling1d_5'))
            model.add(
                Conv1D(100, 100, strides=1, padding='same', kernel_initializer=self.kernel_initializer,
                       name='conv1d_16'))
            model.add(MaxPooling1D(pool_size=10, strides=10, padding='valid', name='max_pooling1d_6'))

        if layers == -1 or layers >= 21:
            model.add(
                Conv1D(4, 5, strides=1, padding='valid', kernel_initializer=self.kernel_initializer, name='conv1d_17'))
            model.add(BatchNormalization(name='batch_normalization_16'))
            model.add(Activation('relu', name='activation_16'))

        model.add(Flatten(name='flatten_1'))

        model.add(Dense(100, kernel_initializer=self.kernel_initializer, name='dense_1'))
        model.add(BatchNormalization(name='batch_normalization_17'))
        model.add(Activation('relu', name='activation_17'))

        model.add(Dense(100, kernel_initializer=self.kernel_initializer, name='dense_2'))
        model.add(BatchNormalization(name='batch_normalization_18'))
        model.add(Activation('relu', name='activation_18'))

        model.add(Dense(5, kernel_initializer=self.kernel_initializer, activation='softmax', name='dense_3'))

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self, layers=-1, previous_model_filename=None):
        model = self.build_model(layers=layers)

        if previous_model_filename:
            if self.verbose > 0:
                print('Loading weights from {}'.format(previous_model_filename))
            model.load_weights(previous_model_filename, by_name=True)

        if self.verbose > 0:
            model.summary()
            print 'Training set:'

        train_x, train_y = unfold(self.train_set, self.verbose)
        class_weight = calculate_weights(self.train_set)
        steps_per_epoch = int(np.math.ceil(1.0 * len(train_y) / self.batch_size))

        if self.verbose > 0:
            print (
                'Layers:{}, Samples:{}, Epochs:{}, Steps:{}'.format(layers, len(train_y), self.epochs, steps_per_epoch))

        name = 'DS_ly{5:d}_e{0:d}-lr{1:g}-dcy{2:g}-m{3:g}-reg{4:g}'.format(self.epochs, self.lr, self.decay, self.m,
                                                                           self.ridge, layers)
        file_path = os.path.join(self.output_dir, name + '_{epoch:03d}-{val_acc:.2f}.h5')
        model_check = ModelCheckpoint(filepath=file_path, monitor='val_loss', verbose=self.verbose, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, min_lr=1e-6, verbose=self.verbose)
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

        if layers == -1:
            filename = os.path.join(self.output_dir, 'model.h5')
            model.save(filename)
            np.savez(os.path.join(self.output_dir, 'history.npz'), acc=history.history['acc'],
                     val_acc=history.history['val_acc'], loss=history.history['loss'],
                     val_loss=history.history['val_loss'])
        else:
            filename = os.path.join(self.output_dir, 'model_layers-{}.h5'.format(layers))
            model.save(filename)
            np.savez(os.path.join(self.output_dir, 'history_layers-{}.npz'.format(layers)), acc=history.history['acc'],
                     val_acc=history.history['val_acc'], loss=history.history['loss'],
                     val_loss=history.history['val_loss'])

        return model, history, filename

    def test_model(self, model, layers=-1):
        # Test model on test set
        if self.verbose > 0:
            if layers == -1:
                print 'Testing model'
            else:
                print 'Testing model with {} layers'.format(layers)

        test_x, y_true = unfold(self.test_data, self.verbose)
        loss_and_metrics = model.evaluate(test_x, y_true, batch_size=self.batch_size, verbose=self.verbose)
        y_pred = model.predict(test_x, batch_size=self.batch_size, verbose=self.verbose)

        if self.verbose > 0:
            print "Layers: {}, Loss: {} Accuracy: {}%".format(layers, loss_and_metrics[0], loss_and_metrics[1] * 100)

        if layers == -1:
            np.savez(os.path.join(self.output_dir, 'test_results.npz'), loss_and_metrics=loss_and_metrics,
                     y_true=y_true, y_pred=y_pred)
        else:
            np.savez(os.path.join(self.output_dir, 'test_results_layer-{}.npz'.format(layers)),
                     loss_and_metrics=loss_and_metrics, y_true=y_true, y_pred=y_pred)

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

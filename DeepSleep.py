#!/bin/python
import os
import glob
import math
import numpy as np
import random
import itertools

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import Constant

random.seed(7493)  # For reproducibility


def next_batch(data, size):
    for item in itertools.cycle(data):
        perm = np.random.permutation(item['Y'].shape[0])
        for i in np.arange(0, item['Y'].shape[0], size):
            yield (item['X'][perm[i:i + size]], item['Y'][perm[i:i + size]])


def unfold(data):
    x, y = np.array(data[0]['X']), np.array(data[0]['Y'])
    for item in data[1:]:
        x = np.concatenate((x, item['X']))
        y = np.concatenate((y, item['Y']))
    return x, y


def count_samples(data):
    return np.sum([item['Y'].shape[0] for item in data])


def count_steps(data, batch_size):
    return int(np.sum([item['Y'].shape[0] / batch_size for item in data]))


def load_data(path):
    """
    Load all npz files from the given path
    :param path: the directory containing npz files
    :return: the data from all the npz files in the given path
    """
    return np.array([np.load(np_name) for np_name in glob.glob(os.path.join(path, '*.np[yz]'))])


def split_data(data, split=0.1):
    """
    Split permutated data into train and test set split by the split value
    :param data: the data
    :param split: the split amount
    :return: the training and test sets
    """
    i = int(len(data) * split)
    perm = np.random.permutation(len(data))  # permute data
    return data[perm[i:]], data[perm[0:i]]  # return training, test sets


def build_model(lr=1e-5, batch_size=64, decay=0., m=0.5, ridge=2e-4, init='he_normal'):
    adam = Adam(lr=lr, decay=decay)
    bias_init = Constant(value=0.1)
    model = Sequential()

    model.add(
        Conv1D(25, 50, padding='valid', input_shape=(15000, 3), kernel_initializer=init, bias_initializer=bias_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(25, 50, padding='valid', kernel_initializer=init, bias_initializer=bias_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling1D())
    model.add(Flatten())

    model.add(Dense(1024, kernel_initializer=init, bias_initializer=bias_init, kernel_regularizer=l2(ridge)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(5, kernel_initializer=init, bias_initializer=bias_init, kernel_regularizer=l2(ridge)))
    model.add(Activation('softmax'))

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def setup(data_dir, k_folds=9, batch_size=192, epochs=100, lr=1e-5, decay=0.9, m=0.5, ridge=2e-4):
    data = load_data(data_dir)
    data, test = split_data(data)
    model = build_model(lr=lr, batch_size=batch_size, decay=decay, m=m, ridge=ridge)
    model.summary()
    fold_size = int(math.ceil(len(data) / k_folds))
    name = 'e' + str(epochs) + '-lr' + str(lr) + '-dcy' + str(decay) + '-mntm' + str(m) + '-reg' + str(ridge)
    filepath = './history/DS_' + name + '_{epoch:03d}-{val_acc:.2f}.h5'
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True)
    earlyStopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    for k in range(k_folds):
        i = int(k * fold_size)
        val = data[i:i + fold_size]
        train = np.concatenate((data[:i], data[i + fold_size:]))
        #         steps_per_epoch = int(count_samples(train) / batch_size)
        steps_per_epoch = count_steps(train, batch_size)
        print count_samples(train), epochs, steps_per_epoch, train.shape, train[0]['Y'].shape

        model.fit_generator(next_batch(train, batch_size), steps_per_epoch, epochs=epochs, verbose=2,
                            validation_data=unfold(val),
                            callbacks=[checkpointer, earlyStopper])


if __name__ == "__main__":
    print 'Setting up'
    setup('/home/afguerrerohernan/data/patients_processed/')


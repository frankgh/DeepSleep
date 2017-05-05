#!/bin/python
#SBATCH -N 1
#SBATCH -o AFG.k1.bn.infront
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:1
#SBATCH -C K80
#SBATCH --mem 64G

import time
start = time.time()

import tensorflow as tf
tf.python.control_flow_ops = tf

import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution1D, Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

import keras.backend
import sys, os
sys.path.append('./')
#from multi_gpu import *

### Model ### 
############################################################
model = Sequential()
reg = 0
mde = 0
filters = 25 # was 25
init = 'he_normal'

model.add(BatchNormalization(mode = mde, input_shape =(15000,3)))
model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid',
                        input_dim=3, input_length=15000, init=init))     ###### change here
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))

model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length = 2, stride=2, border_mode='valid'))

# was just filters for kaggle5
model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length = 2, stride=2, border_mode='valid'))

# was filters * 2 fie kaggle5
model.add(Convolution1D(filters*2, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters*2, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters*2, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length = 2, stride=2, border_mode='valid'))

# was filters*4
model.add(Convolution1D(filters*4, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters*4, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Convolution1D(filters*4, 100, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length = 2, stride=2, border_mode='valid'))
model.add(MaxPooling1D(pool_length = 100, stride=100, border_mode='valid'))
model.add(Convolution1D(4, 5, subsample_length=1, border_mode = 'valid', init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))

# kaggle4 had 36 filters, kaggle5 had 100, kaggle6 had 500
model.add(Flatten())
model.add(Dense(100, init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Dense(100, init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('relu'))
model.add(Dense(6, init=init))
model.add(BatchNormalization(mode = mde))
model.add(Activation('softmax'))

Lr = 1e-5 # was -5
dcy = 0 # was 0
m = 0.5 # kaggle5 is set to 0.5
sgd = SGD(lr=Lr, momentum=m, decay=dcy,  nesterov=True)
model.compile(optimizer = 'sgd',
              loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

### Training - Set up ###
##########################################################

# Loading the Validation Data
# removing Movement class from end

path = './folds/fold1/'
with np.load(path + 'Val.npz') as data:
    yVal = data['Y'][data['Y'][:,-1] == 0]
    xVal = data['X'][data['Y'][:,-1] == 0] #### change here

# Generator Function for Training Data

with np.load(path + 'Train.npz') as data:
    length = data['Y'].shape[0]
    order = np.random.permutation(length)
    
def training_gen(batchsize = 100):
    index = 0
    while True:
        with np.load(path + 'Train.npz') as data:
            xbatch = data['X'][order[index:index+batchsize]] ### change here for different numbers of signals
            ybatch = data['Y'][order[index:index+batchsize]]
        
        yield (xbatch, ybatch)
        del xbatch, ybatch
        index += batchsize
        if index + batchsize > length:
            np.random.shuffle(order)
            index = 0

### Training ###
#########################################################
history = []
epochs = 200
bs = 128 # kaggle5 used to be 128

with np.load(path + 'Train.npz') as data:
    s1 = data['Y'][data['Y'][:,0] == 1].shape[0]
    s2 = data['Y'][data['Y'][:,1] == 1].shape[0]
    s3 = data['Y'][data['Y'][:,2] == 1].shape[0]
    A = data['Y'][data['Y'][:,3] == 1].shape[0]
    R = data['Y'][data['Y'][:,4] == 1].shape[0]

    s1 = float(s2)/s1
    s3 = float(s2)/s3
    A  = float(s2)/A
    R  = float(s2)/R
    s2 /= s2

    print s1, s2, s3, A, R

weight_dict = {0:s1, 1:s2, 2:s3, 3:A, 4:R}
name = '_e' + str(epochs) + '-lr' + str(Lr)+ '-dcy' + str(dcy) + '-mntm' + str(m) + '-reg' + str(reg)

checkpointer = ModelCheckpoint(filepath='./history/k1.bn' + name + '_{epoch:03d}-{val_acc:.2f}.h5', 
                               monitor = 'val_loss', verbose=0, save_best_only=True)



#with tf.device('/gpu:0'):
keras.backend.get_session().run(tf.global_variables_initializer())
a = model.fit_generator(training_gen(batchsize = bs),
                            samples_per_epoch=int(1000/(bs))*bs, # this is a test 
                            nb_epoch=epochs,
                            class_weight = weight_dict,
                            validation_data = (xVal, yVal),
                            callbacks = [checkpointer])

#history.append(a.history)

### Saving Data ###
########################################################

with open('./history/k1.bn' + name + '.pkl', 'wb') as f:
    pickle.dump(a.history, f)

#model.save('./models_history/model_fulldata_bal_' + name + '.h5')

print('time to run in seconds:', time.time() - start)

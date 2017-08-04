# Machine Learning Engineer Nanodegree Capstone Project
# Using artificial neural networks to localize and classify heartbeats in ECG.
#
# Author Piotr Bazan
# August 2nd, 2017
#

from itertools import chain

from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization


def create_encoding_layers(units = [128, 64, 32]):
    return [Dense(u, activation='relu') for u in units]


def create_decoding_layers(units = [64, 128, 784]):
    return [Dense(u, activation='relu' if i < len(units) -1 else 'sigmoid') for i, u in enumerate(units)]


def create_fc_layers(units):
    return [Dense(u, activation='relu' if i < len(units) - 1 else 'softmax') for i, u in enumerate(units)]


def create_encoding_conv_pool_layers(filters = [16, 8, 8], batch_norm = True):
    conv = [Conv2D(f, (3, 3), activation='relu', padding='same') for f in filters]
    pool = [MaxPooling2D((2, 2), padding='same') for f in filters]
    if batch_norm:
        bn = [BatchNormalization() for f in filters]
        return list(chain(*zip(conv, bn, pool)))
    else:
        return list(chain(*zip(conv, pool)))


def create_decoding_conv_pool_layers(filters = [8, 8, 16]):
    conv = [Conv2D(f, (3, 3), activation='relu', \
                   padding='same' if i < len(filters) - 1 else 'valid') for i, f in enumerate(filters)]
    pool = [UpSampling2D((2, 2)) for f in filters]
    last = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    return list(chain(*zip(conv, pool))) + [last]

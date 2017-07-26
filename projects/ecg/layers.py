from itertools import chain
from utils import get_output_shape
import numpy as np

from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Flatten


def create_encoding_layers(units = [128, 64, 32]):
    return [Dense(u, activation='relu') for u in units]


def create_decoding_layers(units = [64, 128, 784]):
    return [Dense(u, activation='relu' if i < len(units) -1 else 'sigmoid') for i, u in enumerate(units)]


def create_fc_layers(units):
    return [Dense(u, activation='relu' if i < len(units) - 1 else 'softmax') for i, u in enumerate(units)]


def create_encoding_conv_pool_layers(kernels = [16, 8, 8], batch_norm = True):
    conv = [Conv2D(k, (3, 3), activation='relu', padding='same') for k in kernels]
    pool = [MaxPooling2D((2, 2), padding='same') for k in kernels]
    if batch_norm:
        bn = [BatchNormalization() for k in kernels]
        return list(chain(*zip(conv, bn, pool)))
    else:
        return list(chain(*zip(conv, pool)))


def create_decoding_conv_pool_layers(kernels = [8, 8, 16]):
    conv = [Conv2D(k, (3, 3), activation='relu', \
                   padding='same' if i < len(kernels) - 1 else 'valid') for i, k in enumerate(kernels)]
    pool = [UpSampling2D((2, 2)) for k in kernels]
    last = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    return list(chain(*zip(conv, pool))) + [last]

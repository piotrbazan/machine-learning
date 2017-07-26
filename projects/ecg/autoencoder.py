from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
from plots import plot_loss_accuracy, plot_ecg

def weights_path(filename):
    return path.join('weights', filename)

def create_encoding_layers(units = [128, 64, 32]):
    return [Dense(u, activation='relu') for u in units]


def create_decoding_layers(units = [64, 128, 784]):
    return [Dense(u, activation='relu' if i < len(units) -1 else 'sigmoid') for i, u in enumerate(units)]


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


def connect_layers(input, layers):
    for i, l in enumerate(layers):
        if i == 0:
            layer = l(input)
        else:
            layer = l(layer)
    return layer


def create_encoders(input_dim = 784, layers_dim = [128, 64], encoding_dim = 32):
    """
    Create NN autoencoder, encoder, decoder
    :param input_dim: 
    :param layers_dim: 
    :param encoding_dim: 
    :return: autoencoder, encoder, decoder
    """
    normal_input  = Input(shape=(input_dim,), name='normal_input')
    encoded_input = Input(shape=(encoding_dim,), name = 'encoded_input')

    encoding_layers = create_encoding_layers(layers_dim + [encoding_dim])
    layers_dim.reverse()
    decoding_layers = create_decoding_layers(layers_dim + [input_dim])

    encoded = connect_layers(normal_input, encoding_layers)
    decoded = connect_layers(encoded, decoding_layers)
    semi_decoded = connect_layers(encoded_input, decoding_layers)

    autoencoder = Model(normal_input, decoded)
    encoder = Model(normal_input, encoded)
    decoder = Model(encoded_input, semi_decoded)

    return autoencoder, encoder, decoder


def create_conv_encoders(input_dim = (28, 28), kernels = [16, 8, 8], encoding_dim = (4, 4)):
    """
    Create convolution autoencoder, encoder, decoder
    :param input_dim: 
    :param kernels: 
    :param encoding_dim: 
    :return: autoencoder, encoder, decoder
    """

    normal_input = Input(shape=input_dim + (1,), name='normal_input')  # adapt this if using `channels_first` image data format
    encoded_input = Input(shape=(encoding_dim + (kernels[-1],)), name = 'encoded_input')

    encoding_layers = create_encoding_conv_pool_layers(kernels)
    kernels.reverse()
    decoding_layers = create_decoding_conv_pool_layers(kernels)

    encoded = connect_layers(normal_input, encoding_layers)
    decoded = connect_layers(encoded, decoding_layers)
    semi_decoded = connect_layers(encoded_input, decoding_layers)

    encoder = Model(normal_input, encoded)
    decoder = Model(encoded_input, semi_decoded)
    autoencoder = Model(normal_input, decoded)

    return autoencoder, encoder, decoder

def get_input_shape(autoencoder):
    return autoencoder.layers[0].input_shape[1:]


def reshape_inputs(x_train, x_test, shape):
    x_train = np.reshape(x_train, (-1,) + shape)
    x_test = np.reshape(x_test, (-1,) + shape)
    return x_train, x_test


def load_weights(model, filename, load_prev):
    if filename is not None and load_prev:
        try:
            model.load_weights(weights_path(filename))
            print('Successfully loaded weights')
        except Exception as e:
            print('Can\'t load weights to model', e)


def save_weights(model, filename):
    if filename is not None:
        try:
            model.save_weights(weights_path(filename))
        except Exception as e:
            print('Can\'t save weights to model', e)


def fit_encoders(encoders, x_train, x_test, epochs=10, filename=None, load_prev=True, verbose = 0):
    autoencoder, encoder, decoder = encoders
    x_train, x_test = reshape_inputs(x_train, x_test, get_input_shape(autoencoder))

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    load_weights(autoencoder, filename, load_prev)
    result = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=512, shuffle=True, verbose=verbose,
                           validation_data=(x_test, x_test))
    save_weights(autoencoder, filename)

    x_decoded = decoder.predict(encoder.predict(x_test))
    plot_diagrams(x_test, x_decoded)
    plot_loss_accuracy(result)
    plot_ecg(x_test[0], x_decoded[0])
    return result

#-----------------------    plotting ---------------------------------------------

def plot_diagrams(x_test, x_decoded):
        n, figsize = 10, (20, 3)
        plt.figure(figsize=figsize)
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_decoded[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        plt.figure(figsize=figsize)
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.plot(x_test[i].reshape((784)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(2, n, i + 1 + n)
            plt.plot(x_decoded[i].reshape((784)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()



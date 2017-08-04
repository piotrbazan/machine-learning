# Machine Learning Engineer Nanodegree Capstone Project
# Using artificial neural networks to localize and classify heartbeats in ECG.
#
# Author Piotr Bazan
# August 2nd, 2017
#

from keras.layers import Flatten, Input, Dropout
from keras.models import Model, Sequential

from layers import *
from plots import plot_loss_accuracy_cnf_matrix, plot_loss_ecg, plot_diagrams
from utils import *


def create_encoders(input_dim = 784, layers_dim = [128, 64], encoding_dim = 32):
    """
    Creates NN autoencoder, encoder, decoder
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


def create_conv_encoders(input_dim = (28, 28), filters = [16, 8, 8], encoding_dim = (4, 4)):   
    """
    Creates convolution autoencoder, encoder, decoder
    :param input_dim: 
    :param filters: 
    :param encoding_dim: 
    :return: autoencoder, encoder, decoder
    """

    normal_input = Input(shape=input_dim + (1,), name='normal_input')  # adapt this if using `channels_first` image data format
    encoded_input = Input(shape=(encoding_dim + (filters[-1],)), name ='encoded_input')

    encoding_layers = create_encoding_conv_pool_layers(filters)
    filters.reverse()
    decoding_layers = create_decoding_conv_pool_layers(filters)

    encoded = connect_layers(normal_input, encoding_layers)
    decoded = connect_layers(encoded, decoding_layers)
    semi_decoded = connect_layers(encoded_input, decoding_layers)

    encoder = Model(normal_input, encoded)
    decoder = Model(encoded_input, semi_decoded)
    autoencoder = Model(normal_input, decoded)

    return autoencoder, encoder, decoder


def create_full_model(encoder, layers_dim = [3]):
    """
    Create full model i.e. encoder + fc layers
    :param encoder: 
    :param layers_dim: 
    :return: full model
    """
    input = Input(shape=get_input_shape(encoder))
    # freeze encoder
    encoder.trainable = False
    # flatten for conv encoder
    layers = [encoder, Flatten()] if len(get_output_shape(encoder)) > 2 else [encoder]
    predictions = connect_layers(input, layers + create_fc_layers(layers_dim))

    return Model(inputs=input, outputs=predictions)


def create_seq_model(filters, units, dropout = 0.):
    conv = [Conv2D(f, (3, 3), activation='relu', padding='same',
                   input_shape = (28, 28, 1) if i == 0 else ()) for i, f in enumerate(filters)]

    pool = [MaxPooling2D((2, 2), padding='same') for f in filters]
    bn = [BatchNormalization() for f in filters]
    conv_layers = list(chain(*zip(conv, bn, pool)))
    dense = [Dense(u, activation='relu') for u in units]
    if dropout:
        dense.append(Dropout(rate = dropout))
    return Sequential(conv_layers + [Flatten()] + dense + [Dense(3, activation='softmax')])


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
    plot_loss_ecg(result, x_test, x_decoded)
    return result


def fit_full_model(model, x_train, x_test, y_train, y_test, classes, epochs = 50, filename=None, load_prev=True, verbose = 1):
    x_train, x_test = reshape_inputs(x_train, x_test, get_input_shape(model))
    y_train, y_test = reshape_inputs(y_train, y_test, get_output_shape(model))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    load_weights(model, filename, load_prev)
    result = model.fit(x_train, y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=verbose,
                             validation_data=(x_test, y_test))
    save_weights(model, filename)

    plot_loss_accuracy_cnf_matrix(result, y_test, model.predict(x_test), classes)
    return result


def predict(model, x):
    x = reshape_input(x, get_input_shape(model))
    return model.predict(x)

from keras.layers import Input, Reshape
from keras.models import Model
from layers import *
from utils import *
import matplotlib.pyplot as plt
from plots import plot_loss_accuracy, plot_ecg


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


def create_conv_encoders(input_dim = (28, 28), kernels = [16, 8, 8], encoding_dim = (4, 4)):
    """
    Creates convolution autoencoder, encoder, decoder
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


def fit(model, x_train, x_test, y_train, y_test,
                   epochs = 50, filename=None, load_prev=True, verbose = 1,
                   optimizer = 'adam', loss = 'mse', metrics = ['accuracy'], batch_size=512):

    x_train, x_test = reshape_inputs(x_train, x_test, get_input_shape(model))
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    load_weights(model, filename, load_prev)
    result = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=verbose,
                             validation_data=(x_test, y_test))
    save_weights(model, filename)

    return result



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



def fit_full_model(model, x_train, x_test, y_train, y_test, epochs = 50, filename=None, load_prev=True, verbose = 1):
    x_train, x_test = reshape_inputs(x_train, x_test, get_input_shape(model))
    y_train, y_test = reshape_inputs(y_train, y_test, get_output_shape(model))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    load_weights(model, filename, load_prev)
    result = model.fit(x_train, y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=verbose,
                             validation_data=(x_test, y_test))
    save_weights(model, filename)

    plot_loss_accuracy(result)
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



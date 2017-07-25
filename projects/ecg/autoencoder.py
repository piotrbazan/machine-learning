from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from itertools import chain

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
    return list(chain(*zip(conv, pool), [last]))


def connect_layers(input, layers):
    for i,l in enumerate(layers):
        if i == 0:
            layer = l(input)
        else:
            layer = l(layer)
    return layer


def create_encoders(input_dim = 784, layers_dim = [128, 64], encoding_dim = 32):
    '''
    Create NN autoencoder, encoder, decoder
    :param input_dim: 
    :param layers_dim: 
    :param encoding_dim: 
    :return: autoencoder, encoder, decoder
    '''
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
    '''
    Create convolution autoencoder, encoder, decoder
    :param input_dim: 
    :param kernels: 
    :param encoding_dim: 
    :return: autoencoder, encoder, decoder
    '''
    normal_input = Input(shape=(*input_dim, 1), name='normal_input')  # adapt this if using `channels_first` image data format
    encoded_input = Input(shape=(*encoding_dim, kernels[-1]), name = 'encoded_input')

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
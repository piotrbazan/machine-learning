import numpy as np
import os.path as path

def weights_path(filename):
    return path.join('weights', filename)


def get_input_shape(model):
    return model.layers[0].input_shape[1:]


def get_output_shape(model):
    return model.layers[-1].output_shape[1:]


def reshape_inputs(x_train, x_test, shape):
    x_train = np.reshape(x_train, (-1,) + shape)
    x_test = np.reshape(x_test, (-1,) + shape)
    return x_train, x_test

def reshape_outputs(y_train, y_test, shape):
    y_train = np.reshape(y_train, (-1,) + shape)
    y_test = np.reshape(y_test, (-1,) + shape)
    return y_train, y_test


def connect_layers(input, layers):
    for i, l in enumerate(layers):
        if i == 0:
            layer = l(input)
        else:
            layer = l(layer)
    return layer


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


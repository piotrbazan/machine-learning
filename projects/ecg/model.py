from keras.layers import Input, Dense
from keras.models import Model

from plots import plot_loss_accuracy


def get_input_shape(encoder):
    return encoder.layers[0].input_shape[1:]

def connect_layers(input, layers):
    for i, l in enumerate(layers):
        if i == 0:
            layer = l(input)
        else:
            layer = l(layer)
    return layer

def create_layers(units):
    return [Dense(u, activation='relu' if i < len(units) - 1 else 'softmax') for i, u in enumerate(units)]


def create_full_model(encoder, layers_dim = [3]):
    x = Input(shape=get_input_shape(encoder))
    # freeze encoder
    encoder.trainable = False
    predictions = connect_layers(x, [encoder] + create_layers(layers_dim))

    model = Model(inputs=x, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fit_full_model(model, x_train, x_test, y_train, y_test, epochs = 50, verbose = 1):
    result = model.fit(x_train, y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=verbose,
                             validation_data=(x_test, y_test))
    plot_loss_accuracy(result)
    return result

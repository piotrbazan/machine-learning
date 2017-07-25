from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

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
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def fit_full_model(model, x_train, x_test, y_train, y_test, epochs = 50, verbose = 1):
    result = model.fit(x_train, y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=verbose,
                             validation_data=(x_test, y_test))

    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.plot(result.history['loss'], label='train loss')
    plt.plot(result.history['val_loss'], label='validation loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(result.history['acc'], label='train acc')
    plt.plot(result.history['val_acc'], label='validation acc')
    plt.legend()


def plot_validation_diagram(model, classes, ann, sig, start, stop):
    plt.figure(figsize=(14, 4))
    plt.plot(sig['MLII'][start:stop])
    a = ann[(ann['Sample'] > start) & (ann['Sample'] < stop)]
    label_pos = start - (stop - start) / 10
    plt.text(label_pos , 0.40, 'Actual', fontsize=12)
    plt.text(label_pos, 0.35, 'Prediction', fontsize=12)
    for i in a.index:
        plt.text(a.loc[i]['Sample'], 0.4, a.loc[i]['Type'], fontsize=12)

    frames = []
    for i in range(start, stop - 784, 10):
        frames.append(np.array(sig['MLII'][i:i + 784]))
    frames = np.array(frames).reshape((-1,) + get_input_shape(model))
    res = model.predict(frames)
    pred = np.argmax(res, axis=1)
    for i, _ in enumerate(frames):
        type = classes[pred[i]]
        if type != 'NB':
            plt.text(start + 784 / 2 + i * 10, 0.35, type, fontsize=12)
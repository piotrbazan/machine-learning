import matplotlib.pyplot as plt
import numpy as np

default_fig_size = (14, 3)


def get_input_shape(model):
    return model.layers[0].input_shape[1:]

def plot_loss_accuracy(result):
    plt.figure(figsize=default_fig_size)
    plt.subplot(1,2,1)
    plt.plot(result.history['loss'], label='train loss')
    plt.plot(result.history['val_loss'], label='validation loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(result.history['acc'], label='train acc')
    plt.plot(result.history['val_acc'], label='validation acc')
    plt.legend()
    plt.show()


def plot_ecg(e1, e2):
    plt.figure(figsize=default_fig_size)
    plt.plot(e1.reshape((784)))
    plt.plot(e2.reshape((784)))
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.show()


def plot_validation_diagram(model, classes, ann, sig, start, stop):
    plt.figure(figsize=default_fig_size)
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

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

default_fig_size = (14, 3)


def get_input_shape(model):
    return model.layers[0].input_shape[1:]


def hide_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_samples(data, beat_types = ['A', 'R', '/', 'V', 'L', 'N'], delta=392):
    for d in data:
        plt.figure(figsize=(14, 2))
        ann = d['annotations']
        sig = d['signals']
        samples = ann[ann['Type'].isin(beat_types)]['Sample']
        for j, sample in enumerate(samples[5:10]):
            plt.subplot(1, 5, 1 + j)
            for signal in sig.columns[1:]:
                signal_window = sig[signal][sample - delta:sample + delta]
                plt.plot(signal_window, label=signal)
            plt.legend()


def plot_avg(signals, sample, delta = 392):
    plt.figure(figsize=(16, 4))
    sig1 = signals['MLII'][sample - delta:sample + delta]
    for j, k in enumerate([3, 5, 7]):
        plt.subplot(1, 3, 1 + j)
        s1 = [np.mean(sig1[i:i + k]) for i, _ in enumerate(sig1)]
        plt.plot(sig1.values)
        plt.plot(s1, label='Avg ' + str(k))
        plt.legend()

def plot_ewma(signals, sample, delta=392):
    plt.figure(figsize=(16, 4))
    sig1 = signals['MLII'][sample - delta:sample + delta]
    for j, k in enumerate([1, 3, 5]):
        plt.subplot(1, 3, 1 + j)
        sig1.plot()
        plt.plot(pd.ewma(sig1, k, adjust=False), label='Ewma ' + str(k))
        plt.legend()


def plot_loss_ecg(result, x_test, x_decoded):
    plt.figure(figsize=default_fig_size)
    ax = plt.subplot(1,2,1)
    ax.set_title('Autoencoder loss')
    plt.plot(result.history['loss'], label='train loss')
    plt.plot(result.history['val_loss'], label='validation loss')
    plt.legend()
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Example of encoded-decoded sample')
    e1, e2 = x_test[10], x_decoded[10]
    plt.plot(e1.reshape((784)))
    plt.plot(e2.reshape((784)))
    hide_axes(ax)
    plt.show()

def plot_loss_accuracy(result):
    plt.figure(figsize=default_fig_size)
    ax = plt.subplot(1,2,1)
    ax.set_title('Full model loss')
    plt.plot(result.history['loss'], label='train loss')
    plt.plot(result.history['val_loss'], label='validation loss')
    plt.legend()
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Full model accuracy')
    plt.plot(result.history['acc'], label='train acc')
    plt.plot(result.history['val_acc'], label='validation acc')
    plt.legend()
    plt.show()

def plot_diagrams(x_test, x_decoded):
    n = 5
    plt.figure(figsize=default_fig_size)
    for i in range(n):
        ax = plt.subplot(2, 2 * n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        hide_axes(ax)
        ax = plt.subplot(2, 2* n, i + 1 + n)
        plt.plot(x_test[i].reshape((784)))
        hide_axes(ax)
        ax = plt.subplot(2, 2 * n, i + 1 + 2* n)
        plt.imshow(x_decoded[i].reshape(28, 28))
        hide_axes(ax)                   
        ax = plt.subplot(2, 2* n, i + 1 + 3*n)
        plt.plot(x_decoded[i].reshape((784)))
        hide_axes(ax)
    plt.show()

    
def plot_validation_diagram(model, classes, ann, sig, start, stop, mark_pred_val = False):
    plt.figure(figsize=default_fig_size)
    plt.plot(sig['MLII'][start:stop])
    
    label_x = start - (stop - start) / 10
    label_y1 = sig['MLII'][start:stop].min() - sig['MLII'][start:stop].ptp() * .3
    label_y2 = sig['MLII'][start:stop].min() - sig['MLII'][start:stop].ptp() * .4
    plt.text(label_x , label_y1, 'Actual', fontsize=12)
    plt.text(label_x, label_y2, 'Prediction', fontsize=12)

    a = ann[(ann['Sample'] > start + 784 / 2) & (ann['Sample'] < stop -784 / 2)]
    for i in a.index:
        plt.text(a.loc[i]['Sample'], label_y1, a.loc[i]['Type'], fontsize=12)

    frames = []
    step = 2
    for i in range(start, stop - 784, step):
        frames.append(np.array(sig['MLII'][i:i + 784]))
    frames = np.array(frames).reshape((-1,) + get_input_shape(model))
    res = model.predict(frames)
    pred_arg, pred_val = np.argmax(res, axis=1), np.max(res, axis=1) 
 
    for i, _ in enumerate(frames):        
        type = classes[pred_arg[i]]
        if type != 'NB':
            color = plt.cm.coolwarm((pred_val[i])) if mark_pred_val else 'black'
            plt.text(start + step * i + 784 / 2, label_y2, type, fontsize=12, color = color)

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.show()

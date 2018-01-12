# Machine Learning Engineer Nanodegree Capstone Project
# Using artificial neural networks to localize and classify heartbeats in ECG.
#
# Author Piotr Bazan
# August 2nd, 2017
#

import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

default_fig_size = (16, 3)
BEAT_TYPES = '?,/,a,A,B,e,E,f,F,j,J,L,n,N,Q,r,R,S,V'.split(',')


def get_input_shape(model):
    return model.layers[0].input_shape[1:]


def hide_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_samples(data, beat_types = ['A', 'R', '/', 'V', 'L', 'N'], delta=392, n = 3):
    for d in data:
        plt.figure(figsize=(11, 2))
        ann = d['annotations']
        sig = d['signals']
        samples = ann[ann['Type'].isin(beat_types)]['Sample']
        for j in range(n):
            try:
                sample = random.choice(samples)
            except KeyError:
                print("Key error")
                continue
            plt.subplot(1, n, 1 + j)
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
        plt.plot(sig1.ewm(com=k,min_periods=0,adjust=False,ignore_na=False).mean(), label='Ewma ' + str(k))
        plt.legend()


def plot_accuracy(result, ax):
    ax.set_title('Full model accuracy')
    plt.plot(result.history['acc'], label='train acc')
    plt.plot(result.history['val_acc'], label='validation acc')
    plt.legend()


def plot_loss(result, ax, title):
    ax.set_title(title)
    plt.plot(result.history['loss'], label='train loss')
    plt.plot(result.history['val_loss'], label='validation loss')
    plt.legend()


def plot_encoded_decoded(encoded, decoded, ax):
    ax.set_title('Example of encoded-decoded sample')
    e1, e2 = encoded[10], decoded[10]
    plt.plot(e1.reshape((784)), label='Original')
    plt.plot(e2.reshape((784)), label='Decoded')
    plt.legend()
    hide_axes(ax)


def plot_loss_ecg(result, x_test, x_decoded):
    plt.figure(figsize=default_fig_size)
    ax = plt.subplot(1,2,1)
    plot_loss(result, ax, 'Autoencoder loss')
    ax = plt.subplot(1, 2, 2)
    plot_encoded_decoded(x_test, x_decoded, ax)
    plt.show()


def plot_cnf_matrix(y_test, y_pred, ax, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    true, pred = np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)
    cm = confusion_matrix(true, pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm > (cm.max() / 2.)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if thresh[i, j] else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # print(precision_recall_fscore_support(true, pred, average='micro'))
#    print(classification_report(true, pred, target_names=classes))


def plot_loss_accuracy_cnf_matrix(result, y_test, y_pred, classes):
    plt.figure(figsize=(14,3))
    ax = plt.subplot(1, 2, 1)
    plot_loss(result, ax, 'Full model loss')
    ax = plt.subplot(1, 2, 2)
    plot_accuracy(result, ax)
    plt.figure(figsize=(13, 6))
    ax = plt.subplot(1, 2, 1)
    plot_cnf_matrix(y_test, y_pred, ax, classes)
    ax = plt.subplot(1, 2, 2)
    plot_cnf_matrix(y_test, y_pred, ax, classes, normalize=True)

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


def plot_validation_diagram(model, classes, ann, sig, start, stop, beat_types = ['N', 'A'], mark_pred_val = False):
    plt.figure(figsize=(16,4))
    plt.plot(sig['MLII'][start:stop])

    label_x = start - (stop - start) / 10
    label_y1 = sig['MLII'][start:stop].min() - sig['MLII'][start:stop].ptp() * .3
    label_y2 = sig['MLII'][start:stop].min() - sig['MLII'][start:stop].ptp() * .4
    plt.text(label_x , label_y1, 'Actual', fontsize=12)
    plt.text(label_x, label_y2, 'Prediction', fontsize=12)

    a = ann[(ann['Sample'] > start + 784 / 2) & (ann['Sample'] < stop -784 / 2)]
    for i in a.index:
        if a.loc[i]['Type'] in beat_types:
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


def plot_validation_diagrams(model, classes, validation, beat_types = ['A', 'L', 'R', 'V']):
    diagrams = 0
    for i, _ in enumerate(validation):
        ann = validation[i]['annotations']
        sig = validation[i]['signals']
        samples = ann[ann['Type'].isin(beat_types)]
        if len(samples) > 0:
            s = samples.sample(1).iloc[0]['Sample']
            plot_validation_diagram(model, classes, ann, sig, s - 3000, s + 3000, mark_pred_val=True, beat_types =BEAT_TYPES)
            diagrams += 1
        if diagrams > 5:
            return


def detect_raw_signal(model, classes, df, start, stop):
    plt.figure(figsize=default_fig_size)
    plt.plot(df['S'][start:stop])
    sig = df['S']

    label_x = start - (stop - start) / 10
    label_y = sig.min() - sig.ptp() * .3
    plt.text(label_x, label_y, 'Prediction', fontsize=12)

    frames = []
    step = 2
    for i in range(start, stop - 784, step):
        array = np.array(sig[i:i + 784])
        frames.append(array)

    frames = np.array(frames).reshape((-1,) + get_input_shape(model))
    res = model.predict(frames)
    pred_arg, pred_val = np.argmax(res, axis=1), np.max(res, axis=1)

    for i, _ in enumerate(frames):
        type = classes[pred_arg[i]]
        if type != 'NB':
            color = plt.cm.coolwarm((pred_val[i]))
            plt.text(start + step * i + 784 / 2, label_y, type, fontsize=12, color=color)
            # print("Frame: %d Type: %s Pred: %.2f Preds: %s" % (step * i + 784 / 2, type, pred_val[i], res[i]))

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.show()

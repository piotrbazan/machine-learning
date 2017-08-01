import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as path
from sklearn.preprocessing import MinMaxScaler

BEAT_TYPES = '?,/,a,A,B,e,E,f,F,j,J,L,n,N,Q,r,R,S,V'.split(',')
#SELECTED_BEAT_TYPES = ['A', 'R', '/', 'V', 'L', 'N']
SELECTED_BEAT_TYPES = ['V', 'N']

def ann_path(i, clean=False):
    if clean:
        return path.join('dataset', '%d.ann.clean.gz' % i)
    else:
        return path.join('dataset', '%d.ann.gz' % i)


def sig_path(i, clean=False):
    if clean:
        return path.join('dataset', '%d.sig.clean.gz' % i)
    else:
        return path.join('dataset', '%d.sig.gz' % i)


def load_data(indices = [0, 1]):
    result = []
    for i in indices:
        ann = pd.read_csv(ann_path(i))
        ann.drop(['Sub', 'Chan', 'Num', 'Aux'], axis = 1, inplace=True)
        sig = pd.read_csv(sig_path(i))
        result.append({'annotations' : ann, 'signals' : sig})
    return result


def load_clean_data(indices=[0, 1], use_cache = True):
    result = []
    for i in indices:
        try:
            if use_cache and path.isfile(ann_path(i, True)):
                ann = pd.read_pickle(ann_path(i, True), compression='gzip')
                sig = pd.read_pickle(sig_path(i, True), compression='gzip')
            else:
                ann = pd.read_csv(ann_path(i))
                ann.drop(['Sub', 'Chan', 'Num', 'Aux'], axis=1, inplace=True)
                sig = pd.read_csv(sig_path(i))
                sig = sig[['sample', 'MLII']]
                scaler = MinMaxScaler()
                sig[['MLII']] = scaler.fit_transform(sig[['MLII']])
                #             s = np.array([np.mean(s[i:i + 5]) for i, _ in enumerate(s)])
                ann.to_pickle(ann_path(i, True), compression='gzip')
                sig.to_pickle(sig_path(i, True), compression='gzip')
        except Exception as e:
            print('Error while parsing file inxed=%d , %s' % (i, str(e)))
        result.append({'annotations': ann, 'signals': sig})
    return np.array(result)


def data_stats(data):
    df = []
    for i, d in enumerate(data):
        d = d['annotations']
        stat = {}
        for t in BEAT_TYPES:
           stat[t] = len(d[d['Type'] == t])
        df.append(stat)
    return pd.DataFrame(df, columns = BEAT_TYPES)    


def safe_random(i, non_beat_margin, delta, size):
    low, high = non_beat_margin
    while True:
        candidate = int(np.random.uniform(-high, high))
        if abs(i - candidate) > low and i + candidate - delta > 0 and i + candidate + delta < size:
            return candidate


def create_features(data, window_size=784, beat_types = SELECTED_BEAT_TYPES):
    """
    Creates features based on annotation data. There are 2 types of data:
    N - normal beat
    A - arrythmia
    :param data: 
    :param window_size: 
    :return: features
    """
    x, delta = [], int(window_size / 2)
    for d in data:
        ann, sig = d['annotations'], d['signals']
        sig_size = len(sig)
        for beat_type in beat_types:
            for i in ann[ann['Type'] == beat_type]['Sample']:
                if delta <= i < sig_size - delta:
                    x.append(np.array(sig['MLII'][i - delta:i + delta]))

    return np.array(x)


def create_features_labels(data, window_size=784, non_beats_per_beat = 9, beat_types = SELECTED_BEAT_TYPES):
    """
    Creates features and labels based on annotation data. There are 3 types of data:
    N - normal beat
    A - arrythmia
    NB - non beat
    :param data: 
    :param window_size: 
    :return: features, labels
    """
    x, y, delta = [], [], int(window_size / 2)
    non_beat_margin = (6, 135)

    for d in data:
        ann, sig = d['annotations'], d['signals']
        sig_size = len(sig)
        for beat_type in beat_types:
            for i in ann[ann['Type'] == beat_type]['Sample']:
                if delta <= i < sig_size - delta:
                    x.append(np.array(sig['MLII'][i - delta:i + delta]))
                    y.append(beat_type)
                    # add non beats
                    for j in range(non_beats_per_beat):
                        c = safe_random(i, non_beat_margin, delta, sig_size)
                        x.append(np.array(sig['MLII'][i + c - delta:i + c + delta]))
                        y.append('NB')

    x_train = np.array(x)
    y_train = np.array(y)
    return x_train, y_train



import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.dummy import  DummyClassifier

from autoencoder import create_full_model, fit_full_model, create_encoders, fit_encoders, create_conv_encoders, \
    create_seq_model, predict
from plots import plot_validation_diagram


def plot_validation(model, classes, ann, sig):
    plot_validation_diagram(model, classes, ann, sig, 147063 - 2000, 147063 + 2500)


def get_stats(model, x_test, y_test, config):
    y_true, y_pred = np.argmax(y_test, axis=1), np.argmax(predict(model, x_test), axis=1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    stats = {
        'model': model,
        'val_acc': accuracy_score(y_true, y_pred),
        'precision': np.mean(p),
        'recall': np.mean(r),
        'f1_score': np.mean(f)
    }
    stats.update(config)
    return stats


def evaluate_dummy(x_train, x_test, y_train, y_test, classes, ann, sig):
    clf = DummyClassifier(random_state=42)
    clf.fit(x_train, y_train)
    y_true, y_pred = np.argmax(y_test, axis=1), np.argmax(clf.predict(x_test), axis=1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    stats = {
        'model': clf,
        'val_acc': accuracy_score(y_true, y_pred),
        'precision': np.mean(p),
        'recall': np.mean(r),
        'f1_score': np.mean(f)
    }
    return stats


def evaluate_full_model(encoder, encoder_name, config, x_train, x_test, y_train, y_test, classes, ann, sig, epochs=2, load_prev = False):
    stats = []
    for layers_dim in config:
        model_name = str(layers_dim)
        print('Evaluating model with fc:', model_name)
        model = create_full_model(encoder, layers_dim)
        fit_full_model(model, x_train, x_test, y_train, y_test, classes, epochs=epochs, filename=encoder_name + model_name, load_prev=load_prev, verbose=0)

        plot_validation(model, classes, ann, sig)
        stats.append(get_stats(model, x_test, y_test, {'ae': encoder_name, 'fc': model_name}))
    return stats


def evaluate_seq_models(config, x_train, x_test, y_train, y_test, classes, ann, sig, epochs=1, load_prev=True):
    stats = []
    for c in config:
        model_name = str(c)
        print('Running sequential model:', model_name)
        model = create_seq_model(filters=c['filters'], units=c['units'], dropout=c['dropout'])
        fit_full_model(model, x_train, x_test, y_train, y_test, classes, epochs=epochs, filename=model_name + '.h5', load_prev=load_prev, verbose=1)

        plot_validation(model, classes, ann, sig)
        stats.append(get_stats(model, x_test, y_test, c))

    return stats


def evaluate_ae_models(config, x_train, x_test, y_train, y_test, classes, ann, sig, ae_epochs=1, full_model_epochs=1, load_prev_ae=True, load_prev_full=True):
    result = []
    create_encoders_func = create_conv_encoders if config['use_conv'] else create_encoders
    for cfg in config['ae']:
        ae_name = str(cfg)
        encoders = create_encoders_func(*cfg)
        print('Running autoencoder with config:', ae_name)
        fit_encoders(encoders, x_train, x_test, epochs=ae_epochs, filename=ae_name + '.h5', load_prev=load_prev_ae, verbose=0)

        result.extend(evaluate_full_model(encoders[1], ae_name, config['fc'], x_train, x_test, y_train, y_test, classes, ann, sig, epochs=full_model_epochs, load_prev=load_prev_full))

    return result

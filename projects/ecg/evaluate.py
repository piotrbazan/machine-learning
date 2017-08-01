import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from autoencoder import create_full_model, fit_full_model, create_encoders, fit_encoders, create_conv_encoders, \
    create_seq_model, predict
from plots import plot_validation_diagram


def plot_validation(model, classes, ann, sig):
    plot_validation_diagram(model, classes, ann, sig, 144425 - 2000, 144425 + 2500)


def evaluate_full_model(encoder, encoder_name, config, x_train, x_test, y_train, y_test, classes, ann, sig, load_prev, epochs=2):
    result = []
    for layers_dim in config:
        model_name = str(layers_dim)
        print('Evaluating model with fc:', model_name)
        model = create_full_model(encoder, layers_dim)
        h = fit_full_model(model, x_train, x_test, y_train, y_test, classes, epochs=epochs, filename=encoder_name + model_name, load_prev=load_prev, verbose=0)

        y_true, y_pred = np.argmax(y_test, axis=1), np.argmax(predict(model, x_test), axis=1)
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        result.append({
            'model': model,
            'ae': encoder_name,
            'fc': model_name,
            'val_acc': h.history['val_acc'][-1],
            'precision': np.mean(p),
            'recall': np.mean(r),
            'f1_score': np.mean(f)
        })
        plot_validation(model, classes, ann, sig)

    return result


def evaluate_nn_models(config, x_train, x_test, y_train, y_test, classes, ann, sig, ae_epochs=1, full_model_epochs=1, load_prev_ae=True, load_prev_full=True):
    result = []
    for layers_dim in config['ae']:
        ae_name = str(layers_dim)
        print('Running autoencoder with config:', ae_name)
        encoders = create_encoders(*layers_dim)
        fit_encoders(encoders, x_train, x_test, epochs=ae_epochs, filename=ae_name + '.h5', load_prev=load_prev_ae, verbose=0)

        result.extend(evaluate_full_model(encoders[1], ae_name, config['fc'], x_train, x_test, y_train, y_test, classes, ann, sig, load_prev_full, full_model_epochs))

    return result


def evaluate_conv_models(config, x_train, x_test, y_train, y_test, classes, ann, sig, ae_epochs=1, full_model_epochs=1, load_prev_ae=True, load_prev_full=True):
    result = []
    ae_name = 'conv_16_8_8'
    print('Running convolution autoencoder:', ae_name)
    encoders = create_conv_encoders()
    fit_encoders(encoders, x_train, x_test, epochs=ae_epochs, filename=ae_name + '.h5', load_prev=load_prev_ae, verbose=0)

    result.extend(evaluate_full_model(encoders[1], ae_name, config['fc'], x_train, x_test, y_train, y_test, classes, ann, sig, load_prev_full, epochs=full_model_epochs))
    return result


def evaluate_seq_models(config, x_train, x_test, y_train, y_test, classes, ann, sig, epochs=1, load_prev=True):
    models = []
    for c in config:
        model_name = str(c)
        print('Running sequential model:', model_name)
        model = create_seq_model(filters=c['filters'], units=c['units'], dropout=c['dropout'])
        h = fit_full_model(model, x_train, x_test, y_train, y_test, classes, epochs=epochs, filename=model_name + '.h5', load_prev=load_prev, verbose=1)
        models.append({'model': model, 'val_acc': h.history['val_acc'][-1], 'config': c})
        plot_validation(model, classes, ann, sig)

    return models

import numpy as np
import pickle
import matplotlib.pyplot as plt


def evaluate_prediction(predicted, label):
    labels = np.asarray(label).squeeze()
    n_feature, n_data = labels.shape
    single_feat_pred_right = []
    acc = []
    for i in range(0, n_feature):
        val = np.argmax(predicted[i], axis=1).astype('uint8')
        single_feat_pred_right.append(np.count_nonzero(val == labels[i].flatten()) / n_data * 100)
        acc.append(val)

    out_y_array = np.asarray(acc).T
    seq_acc = np.count_nonzero(np.all(out_y_array[:, 1:5] == labels[1:5, :].T, axis=1)) / float(n_data) * 100
    return {'acc': single_feat_pred_right, 'predicted_y': acc, 'seq_acc': seq_acc}


def evaluate_model_performance(fitted_model, model_name, data, model):
    X_train = data["X_train"]
    X_test  = data["X_test"]
    X_val  = data["X_val"]
    y_train = list(data["y_train"].T)
    y_test  = list(data["y_test"].T)
    y_val  = list(data["y_val"].T)

    for i in ['1', '2', '3', '4']:
        show_and_save_fig(fitted_model.history['dig{}_acc'.format(i)],
                          fitted_model.history['val_dig{}_acc'.format(i)],
                          title='Digit{} Acc'.format(i),
                          ylabel='accuracy',
                          model_name=model_name)

    for i in ['1', '2', '3', '4']:
        show_and_save_fig(fitted_model.history['dig{}_loss'.format(i)],
                          fitted_model.history['val_dig{}_loss'.format(i)],
                          title='Digit{} Loss'.format(i),
                          ylabel='loss',
                          model_name=model_name)

    show_and_save_fig(fitted_model.history['num_acc'],
                      fitted_model.history['val_num_acc'],
                      title='Digit Number Acc',
                      ylabel='accuracy',
                      model_name=model_name)

    show_and_save_fig(fitted_model.history['loss'],
                      fitted_model.history['val_loss'],
                      title='Model Loss',
                      ylabel='loss',
                      model_name=model_name)

    show_and_save_fig(fitted_model.history['nC_acc'],
                      fitted_model.history['val_nC_acc'],
                      title='Digit Classifier accuracy',
                      ylabel='accuracy',
                      model_name=model_name)
    train_acc, train_score, train_seq_acc = print_final_score(model, X_train, y_train, set_name='Train')
    val_acc, val_score, val_seq_acc = print_final_score(model, X_val, y_val, set_name='Validation')
    test_acc, test_score, test_seq_acc = print_final_score(model, X_test, y_test, set_name='Test')

    metrics = {'train_acc': train_acc,
               'test_acc': test_acc,
               'val_acc': val_acc,
               'train_seq_acc': train_seq_acc,
               'test_seq_acc': test_seq_acc,
               'val_seq_acc': val_seq_acc,
               'train_score': train_score,
               'test_score': test_score,
               'val_score': val_score}
    with open('metrics/' + model_name + '.pickle', 'wb') as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('metrics/' + model_name + 'History.pickle', 'wb') as f:
        pickle.dump(fitted_model.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    return metrics


def show_and_save_fig(x, x2, title, ylabel, model_name):
    fig1 = plt.gcf()
    plt.ylim([0, 1])
    plt.plot(x, c='k')
    plt.plot(x2, c='r')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.draw()
    fig1.savefig('plots/{}_'.format(title.replace(' ', '_')) + model_name + '.png', bbox_inches='tight', dpi=200)
    plt.close()


def print_final_score(model, X, y, set_name):
    y_predicted = model.predict(X)
    score = model.evaluate(X, y, verbose=0)
    metrics = evaluate_prediction(y_predicted,
                                  y)
    acc = metrics['acc']
    seq_acc = metrics['seq_acc']
    print('{} loss:'.format(set_name), score[0])
    print('number_of_digits', 'digit1', 'digit2', 'digit3', 'digit4')
    print('{} accuracy:'.format(set_name), acc)
    print('{} sequence accuracy:'.format(set_name), seq_acc)
    return acc, score, seq_acc

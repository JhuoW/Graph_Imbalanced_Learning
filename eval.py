import numpy as np
import functools

from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(3)
def label_classification(args, data, embeddings, ratio):
    data = data.detach().cpu()
    y = data.y
    
    new_y = data.new_y
    # occurrences_dict = {}
    # new_y = y[data.imb_train_mask].numpy()
    # for number in new_y:
    #     occurrences_dict[number] = occurrences_dict.get(number, 0) + 1

    # print(occurrences_dict)

    

    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    new_Y = new_y.detach().cpu().numpy()
    new_Y = new_Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(new_Y)
    new_Y = onehot_encoder.transform(new_Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    if args.split == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio)
    elif args.split == 'public':
        train_mask = data.train_mask
        test_mask  = data.test_mask
        X_train = X[train_mask]
        X_test  = X[test_mask]
        y_train = Y[train_mask]
        y_test  = Y[test_mask]
    elif args.split == 'imbalance':
        train_mask = data.imb_train_mask
        test_mask  = data.test_mask
        X_train = X[train_mask]
        X_test  = X[test_mask]
        y_train = new_Y[train_mask]
        y_test  = Y[test_mask]


    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)  # (1000, 7)

    # preds = np.where(y_pred == True)[1]

    acc = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    # acc   = (data.y[test_mask].numpy() == preds).sum().item()/test_mask.sum().item()

    # y_test: (1000, 7),     y_pred:(1000, 7)
    bacc   = balanced_accuracy_score(np.where(y_test == True)[1], np.where(y_pred == True)[1])  # 

    return {
        'Acc': acc,
        'Macro-F1': macro,
        'BAcc': bacc
        # 'acc' : acc
    }

# 
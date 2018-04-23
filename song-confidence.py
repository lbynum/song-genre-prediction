import numpy as np
import matplotlib.pyplot as plt

from util import MSDMXMData, stratified_random_sample_MXMMSD, select_genres_MXMMSD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def get_other_metrics():
    data = MSDMXMData()
    data.load_data()

    data = stratified_random_sample_MXMMSD(
        data,
        sample_proportion=0.1,
        random_state=123)

    # select only two genres
    genre_list = ['Rap', 'Pop']
    data = select_genres_MXMMSD(data, genre_list)

    print("X_train:", data.X_train.shape)
    print("y_train:", data.y_train.shape)
    print("X_test:", data.X_test.shape)
    print("y_test:", data.y_test.shape)
    # encode labels to get rid of strings
    data.encode_labels()

    dummy_clf = DummyClassifier(strategy='most_frequent')
    rf_clf = RandomForestClassifier(class_weight='balanced', criterion='gini',
                                    max_depth=None, max_features='sqrt',
                                    n_estimators=50)
    log_reg_clf = LogisticRegression(C=1, class_weight='balanced')
    linear_svm_clf = SVC(kernel='linear', C=1, class_weight='balanced')
    rbf_svm_clf = SVC(kernel='rbf', C=1, class_weight='balanced', gamma='auto')

    clf_names = ['Dummy', 'Random Forest', 'Logistic Regression', 'Linear SVM', 'RBF SVM']
    clf_list = [dummy_clf, rf_clf, log_reg_clf, linear_svm_clf, rbf_svm_clf]

    y_train = (data.y_train == 10).astype(int)
    y_test = (data.y_test == 10).astype(int)

    # metrics = ['accuracy', 'f1_score', 'auroc', 'recall', 'precision']
    # score_tuples = [[], [], [], [], []]

    # print('BOOTSTRAPPING')
    # num_bootstrap_samples = 1000
    # n, _ = data.X_test.shape
    # bootstrap_indices = np.random.randint(0, n, (num_bootstrap_samples, n))

    plt.hist(y_test)
    plt.show()

    linear_svm_clf.fit(data.X_train, y_train)
    y_pred = linear_svm_clf.decision_function(data.X_test)
    max_index = np.argmax(y_pred)
    min_index = np.argmin(y_pred)
    # print('Max index', max_index)

    rap_probabilities = y_pred# - 0.5
    min_rap_prob = 0#rap_probabilities[0]
    min_rap_index = 0
    for i, prob in enumerate(rap_probabilities):
        if prob >= 0 and prob <= min_rap_prob:
            min_rap_prob = prob
            min_rap_index = i

    pop_probabilities = y_pred# + 0.5
    min_pop_prob = -1000#pop_probabilities[0]
    min_pop_index = 0
    for i, prob in enumerate(pop_probabilities):
        if prob <= 0 and prob >= min_pop_prob:
            min_pop_prob = prob
            min_pop_index = i


    print('Most Confident Rap Song', y_pred[max_index], data.TID_test[max_index])
    print('Least Confident Rap Song', y_pred[min_rap_index], data.TID_test[min_rap_index])
    print('Most Confident Pop Song', y_pred[min_index], data.TID_test[min_index])
    print('Least Confident Pop Song', y_pred[min_pop_index], data.TID_test[min_pop_index])

    # for i, clf in enumerate(clf_list):
    #     print(clf_names[i])
    #     clf.fit(data.X_train, y_train)
    #     y_pred = clf.predict_proba(data.X_test)
    #
    #


        # print(y_pred)
        # print(y_test)
        # print("\t\tACCURACY:", accuracy_score(y_test, y_pred))
        # print("\t\tF1 SCORE:", f1_score(y_test, y_pred, average='weighted'))
        # print("\t\tROC AUC:", roc_auc_score(y_test, y_pred, average='weighted'))
        # print("\t\tRECALL:", recall_score(y_test, y_pred, average='weighted'))
        # print("\t\tPRECISION:", precision_score(y_test, y_pred, average='weighted'))


get_other_metrics()
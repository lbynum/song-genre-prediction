from collections import Counter
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, recall_score,
                             precision_score, make_scorer)

from util import (MusixMatchData, stratified_random_sample_mxm,
                  select_genres_mxm, select_genres_MXMMSD, MSDMXMData,
                  stratified_random_sample_MXMMSD)

def run_msd():
    ############################################################################
    # load data
    ############################################################################
    data = MSDMXMData()
    data.load_data()

    data = stratified_random_sample_MXMMSD(
        data,
        sample_proportion=0.1,
        random_state=123)

    # select only two genres
    genre_list = ['Rap', 'Pop']
    data = select_genres_MXMMSD(data, genre_list)

    # encode labels to get rid of strings
    data.encode_labels()

    # select only msd features
    colnames = data.colnames[-10:]
    X_train = data.X_train[:,-10:]
    y_train = data.y_train

    X_test = data.X_test[:,-10:]
    y_test = data.y_test







def run_musixmatch():
    ############################################################################
    # train models
    ############################################################################
    data_train = MusixMatchData()
    try:
        data_train.load_from_pickle(pickled_data_path='data/musixmatch/pickled/',
                              suffix='train')
    except:
        data_train.write_to_pickle(
            X_filename='data/musixmatch/mxm_dataset_train.txt',
            genre_filename='genres.csv',
            pickled_data_path='data/musixmatch/pickled/',
            suffix='train'
        )

    data_train = stratified_random_sample_mxm(
            data_train,
            sample_proportion=0.1,
            random_state=123)

    # select only two genres
    genre_list = ['Rap', 'Pop']
    data_train = select_genres_mxm(data_train, genre_list)

    # encode labels to get rid of strings
    data_train.encode_labels()

    # print quick summary statistics
    # print(Counter(data.y))
    # print(len(data.X))

    # plot the classes
    # plt.hist(data.y)
    # plt.show()

    pipelines_dict = {
        'DummyClassifier':
            Pipeline([
                ('clf', DummyClassifier())
            ]),
        'RandomForestClassifier':
            Pipeline([
                ('clf', RandomForestClassifier())
            ]),
        'linear_SVM':
            Pipeline([
                ('clf', SVC(kernel='linear'))
            ]),
        # 'LogisticRegression':
        #     Pipeline([
        #         ('clf', LogisticRegression())
        #     ]),
    }

    parameters_dict = {
        'DummyClassifier':
            {
                'clf__strategy': ('most_frequent',)
                #'('stratified', 'most_frequent', 'prior', 'uniform')
            },
        'RandomForestClassifier':
            {
                'clf__n_estimators': (10,)
            },
        'LogisticRegression':
            {
                'clf__C': (1,),#tuple(10.0 ** np.arange(-3, 3)),
                'clf__class_weight': ('balanced', None)
            },
        'linear_SVM':
            {
                'clf__C': (1,),  # tuple(10.0 ** np.arange(-3, 3)),
                # 'clf__class_weight': ('balanced', None)
            },
    }

    best_estimators = defaultdict(dict)


    # grid search with respect to different metrics and print results
    # define scorers for multi-class classification
    accuracy = make_scorer(accuracy_score)
    precision = make_scorer(precision_score, average='weighted')
    f1 = make_scorer(f1_score, average='weighted')
    recall = make_scorer(recall_score, average='weighted')
    roc_auc = make_scorer(roc_auc_score, average='weighted')
    scoring_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'recall': recall,
        'roc_auc': roc_auc,
    }

    print(
    '''
    ############################################################################
    # Grid Search CV Performance
    ############################################################################
    '''
    )

    # loop through each classifier and each metric to get CV performance
    for clf_name, pipeline in pipelines_dict.items():
        parameters = parameters_dict[clf_name]
        for metric_name, metric in scoring_dict.items():
            if (clf_name == 'DummyClassifier'
                and metric_name in ['f1', 'precision', 'recall']):
                # skip ill-defined metrics
                continue

            grid_search = GridSearchCV(
                pipeline,
                parameters,
                scoring=metric,
                n_jobs=-2,
                verbose=0
            )
            grid_search.fit(data_train.X, data_train.y)

            # store best estimator in dict
            best_estimator = grid_search.best_estimator_
            best_estimators[clf_name][metric_name] = best_estimator

            # print params for best fit
            print("\tBest {} with {}: {}".format(clf_name,
                                            metric,
                                            grid_search.best_score_))
            print("\t\tBest parameters set:")
            best_parameters = best_estimator.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t\t\t%s: %r" % (param_name, best_parameters[param_name]))


    ############################################################################
    # predict on test data
    ############################################################################
    data_test = MusixMatchData()
    try:
        data_test.load_from_pickle(pickled_data_path='data/musixmatch/pickled/',
                              suffix='test')
    except:
        data_test.write_to_pickle(
            X_filename='data/musixmatch/mxm_dataset_test.txt',
            genre_filename='genres.csv',
            pickled_data_path='data/musixmatch/pickled/',
            suffix='test'
        )

    # select the same genres
    data_test = select_genres_mxm(data_test, genre_list)
    data_test.encode_labels()

    print(
    '''
    ############################################################################
    # Test Performance
    ############################################################################
    '''
    )

    # predict using each best estimator (one for each metric)
    for clf_name in best_estimators.keys():
        print('\n\tClassifier: {}'.format(clf_name))
        entry = best_estimators[clf_name]
        for metric_name, estimator in entry.items():
            if (clf_name == 'DummyClassifier'
                and metric_name in ['f1', 'precision', 'recall']):
                # skip ill-defined metrics
                continue

            y_pred = estimator.predict(data_test.X)
            y_true = data_test.y
            matrix = confusion_matrix(
                y_true=y_true,
                y_pred=y_pred
            )
            # print confusion matrix and score
            print('\tBest Estimator Performance -- {}:'.format(metric_name))
            print('\t\t' + str(matrix).replace('\n', '\n\t\t'))
            scorer = scoring_dict[metric_name]
            score = scorer(estimator, data_test.X, data_test.y)
            print('\t\t{}: {}'.format(metric_name, score))

    ############################################################################
    # explore important features
    ############################################################################
    # RF
    print(
    '''
    ############################################################################
    # Random Forest Feature Importance
    ############################################################################
    '''
    )
    best_pipeline = best_estimators['RandomForestClassifier']['accuracy']

    importance_scores = best_pipeline._final_estimator.feature_importances_

    feature_importances = list(zip(data_train.vocab, importance_scores))
    feature_importances = sorted(feature_importances,
                                 key=lambda pair: pair[1],
                                 reverse=True)
    num_features = 20
    # print top 20
    print('\tTop {} Features for {} vs {}'.format(num_features,
                                                  genre_list[0],
                                                  genre_list[1]))
    for pair in feature_importances[:num_features]:
        print('\t\t{}'.format(pair))

    # SVM
    print(
    '''
    ############################################################################
    # Linear SVM Feature Importance
    ############################################################################
    '''
    )
    best_pipeline = best_estimators['linear_SVM']['accuracy']

    importance_scores = list(best_pipeline._final_estimator.coef_[0])

    feature_importances = list(zip(data_train.vocab, importance_scores))
    feature_importances = sorted(feature_importances,
                                 key=lambda pair: pair[1],
                                 reverse=True)
    num_features = 20
    # print top 20
    print('\tTop {} Features for {}'.format(num_features, genre_list[1]))
    for pair in feature_importances[:num_features]:
        print('\t\t{}'.format(pair))

    # print bottom 20
    print('\tTop {} Features for {}'.format(num_features, genre_list[0]))
    for pair in feature_importances[-num_features:]:
        print('\t\t{}'.format(pair))


def run_all_data():
    ############################################################################
    # train models
    ############################################################################
    data = MSDMXMData()
    data.load_data()

    data = stratified_random_sample_MXMMSD(
        data,
        sample_proportion=0.1,
        random_state=123)

    # select only two genres
    genre_list = ['Rap', 'Pop']
    data = select_genres_MXMMSD(data, genre_list)

    # encode labels to get rid of strings
    data.encode_labels()
    print(set(data.y_train))

    # print quick summary statistics
    # print(Counter(data.y))
    # print(len(data.X))

    # plot the classes
    # plt.hist(data.y)
    # plt.show()

    pipelines_dict = {
        'DummyClassifier':
            Pipeline([
                ('clf', DummyClassifier())
            ]),
        'RandomForestClassifier':
            Pipeline([
                ('clf', RandomForestClassifier())
            ]),
        'linear_SVM':
            Pipeline([
                ('clf', SVC(kernel='linear'))
            ]),
        # 'LogisticRegression':
        #     Pipeline([
        #         ('clf', LogisticRegression())
        #     ]),
    }

    parameters_dict = {
        'DummyClassifier':
            {
                'clf__strategy': ('most_frequent',)
                #'('stratified', 'most_frequent', 'prior', 'uniform')
            },
        'RandomForestClassifier':
            {
                'clf__n_estimators': (10,)
            },
        'LogisticRegression':
            {
                'clf__C': (1,),#tuple(10.0 ** np.arange(-3, 3)),
                'clf__class_weight': ('balanced', None)
            },
        'linear_SVM':
            {
                'clf__C': (1,),  # tuple(10.0 ** np.arange(-3, 3)),
                # 'clf__class_weight': ('balanced', None)
            },
    }

    best_estimators = defaultdict(dict)


    # grid search with respect to different metrics and print results
    # define scorers for multi-class classification
    accuracy = make_scorer(accuracy_score)
    precision = make_scorer(precision_score, average='weighted')
    f1 = make_scorer(f1_score, average='weighted')
    recall = make_scorer(recall_score, average='weighted')
    roc_auc = make_scorer(roc_auc_score, average='weighted')
    scoring_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'recall': recall,
        # 'roc_auc': roc_auc,
    }

    print(
    '''
    ############################################################################
    # Grid Search CV Performance
    ############################################################################
    '''
    )

    # loop through each classifier and each metric to get CV performance
    for clf_name, pipeline in pipelines_dict.items():
        parameters = parameters_dict[clf_name]
        for metric_name, metric in scoring_dict.items():
            if (clf_name == 'DummyClassifier'
                and metric_name in ['f1', 'precision', 'recall']):
                # skip ill-defined metrics
                continue

            grid_search = GridSearchCV(
                pipeline,
                parameters,
                scoring=metric,
                n_jobs=1,
                verbose=0
            )
            grid_search.fit(data.X_train, data.y_train)

            # store best estimator in dict
            best_estimator = grid_search.best_estimator_
            best_estimators[clf_name][metric_name] = best_estimator

            # print params for best fit
            print("\tBest {} with {}: {}".format(clf_name,
                                            metric,
                                            grid_search.best_score_))
            print("\t\tBest parameters set:")
            best_parameters = best_estimator.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t\t\t%s: %r" % (param_name, best_parameters[param_name]))


    ############################################################################
    # predict on test data
    ############################################################################
    print(
    '''
    ############################################################################
    # Test Performance
    ############################################################################
    '''
    )

    # predict using each best estimator (one for each metric)
    for clf_name in best_estimators.keys():
        print('\n\tClassifier: {}'.format(clf_name))
        entry = best_estimators[clf_name]
        for metric_name, estimator in entry.items():
            if (clf_name == 'DummyClassifier'
                and metric_name in ['f1', 'precision', 'recall']):
                # skip ill-defined metrics
                continue

            y_pred = estimator.predict(data.X_test)
            y_true = data.y_test
            matrix = confusion_matrix(
                y_true=y_true,
                y_pred=y_pred
            )
            # print confusion matrix and score
            print('\tBest Estimator Performance -- {}:'.format(metric_name))
            print('\t\t' + str(matrix).replace('\n', '\n\t\t'))
            scorer = scoring_dict[metric_name]
            score = scorer(estimator, data.X_test, data.y_test)
            print('\t\t{}: {}'.format(metric_name, score))

    ############################################################################
    # explore important features
    ############################################################################
    # RF
    print(
    '''
    ############################################################################
    # Random Forest Feature Importance
    ############################################################################
    '''
    )
    best_pipeline = best_estimators['RandomForestClassifier']['accuracy']

    importance_scores = best_pipeline._final_estimator.feature_importances_

    feature_importances = list(zip(data.colnames, importance_scores))
    feature_importances = sorted(feature_importances,
                                 key=lambda pair: pair[1],
                                 reverse=True)
    num_features = 20
    # print top 20
    print('\tTop {} Features for {} vs {}'.format(num_features,
                                                  genre_list[0],
                                                  genre_list[1]))
    for pair in feature_importances[:num_features]:
        print('\t\t{}'.format(pair))

    # SVM
    print(
    '''
    ############################################################################
    # Linear SVM Feature Importance
    ############################################################################
    '''
    )
    best_pipeline = best_estimators['linear_SVM']['accuracy']

    importance_scores = list(best_pipeline._final_estimator.coef_[0])

    feature_importances = list(zip(data.colnames, importance_scores))
    feature_importances = sorted(feature_importances,
                                 key=lambda pair: pair[1],
                                 reverse=True)
    num_features = 20
    # print top 20
    print('\tTop {} Features for {}'.format(num_features, genre_list[1]))
    for pair in feature_importances[:num_features]:
        print('\t\t{}'.format(pair))

    # print bottom 20
    print('\tTop {} Features for {}'.format(num_features, genre_list[0]))
    for pair in feature_importances[-num_features:]:
        print('\t\t{}'.format(pair))


if __name__ == '__main__':
    # run_all_data()
    # run_musixmatch()
    run_msd()
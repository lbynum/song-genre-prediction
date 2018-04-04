import numpy as np
from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from util import MusixMatchData


def main():
    data = MusixMatchData()
    try:
        data.load_from_pickle()
    except:
        data.write_to_pickle(
            X_filename='data/musixmatch/mxm_dataset_train.txt',
            genre_filename='genres.csv'
        )

    parameters = {
        'clf__C': tuple(10.0 ** np.arange(-3, 3))
    }

    pipeline = Pipeline([
        ('clf', LogisticRegression())
    ])


    # grid search with respect to different metrics and print results
    scoring = ['accuracy', 'precision', 'f1', 'recall', 'roc_auc']
    for metric in scoring:
        grid_search = GridSearchCV(
            pipeline,
            parameters,
            scoring=metric,
            n_jobs=-2,
            verbose=3
        )

        grid_search.fit(data.X, data.y)

        # print params for best fit
        print("Best {}: {}".format(metric, grid_search.best_score_))
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    main()
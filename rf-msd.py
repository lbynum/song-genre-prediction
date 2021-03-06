import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from util import MSDData


def main():
    data = MSDData()
    data.load_data()

    # Compare two genres against each other
    # data.load_data('Blues', 'Rap')

    X = data.X
    y = data.y
    n, d = X.shape

    index_array = np.arange(n)
    sample_indices, _ = train_test_split(index_array, stratify=data.y,
                                         train_size=0.8, random_state=123)

    # select examples from split
    X_train = data.X[sample_indices]
    y_train = data.y[sample_indices]
    TID_train = data.TID[sample_indices]

    parameters = {
        'clf__max_features': tuple(np.arange(1, d+1)),
        'clf__n_estimators': (10, 50, 100),
        'clf__criterion': ('gini', 'entropy'),
        'clf__max_depth': (5, 10, 20)
        # 'clf__C': 10**np.arange(-3, 3, dtype=float),
        # 'clf__class_weight': ('balanced', None)
        # 'clf__strategy': ('stratified', 'most_frequent', 'prior', 'uniform')
    }

    pipeline = Pipeline([
        ('clf', RandomForestClassifier())
    ])

    # grid search with respect to different metrics and print results
    scoring = ['accuracy']#, 'precision', 'f1', 'recall', 'roc_auc']
    for metric in scoring:
        grid_search = GridSearchCV(
            pipeline,
            parameters,
            scoring=metric,
            n_jobs=-2,
            verbose=3,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        # print params for best fit
        print("Best {}: {}".format(metric, grid_search.best_score_))
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        print(grid_search.cv_results_)

if __name__ == '__main__':
    main()
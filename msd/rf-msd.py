import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

class MSDData:
    '''
    Class for storing msd data.
    '''
    def __init__(self, X=None, y=None, TID=None):
        '''
        Data class.

        Attributes
        --------------------
        X       -- numpy array of shape (n,d), features
        y       -- numpy array of shape (n,1), labels
        TID     -- numpy array of shape (n,1), track IDs


        '''
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
        self.TID = TID

    def load_data(self):
        '''
            Load csv file into X, y, TID -- removes any examples with missing 
            features

        '''
        # grab songs from csv files, remove any with missing features
        msd_csv_path = 'features_vs_genre.csv'
        df = pd.read_csv(msd_csv_path)
        df = df[df.year != 0].dropna(axis=0, how='any')
        TID = df.as_matrix(columns=[df.columns[0]])
        X = df.as_matrix(columns=df.columns[2:])
        y = df.as_matrix(columns=[df.columns[1]])

        # convert genre labels to numerical representation
        le = LabelEncoder()
        le.fit(y.ravel())
        y = le.transform(y.ravel())

        self.X = X
        self.y = y
        self.TID = TID

def main():
    data = MSDData()
    data.load_data()

    X = data.X
    y = data.y
    n, d = X.shape

    index_array = np.arange(n)
    sample_indices, _ = train_test_split(index_array, stratify=data.y,
                                         train_size=0.1, random_state=123)

    # select examples from split
    X_train = data.X[sample_indices]
    y_train = data.y[sample_indices]
    TID_train = data.TID[sample_indices]

    parameters = {
        'clf__max_features': tuple(np.arange(1, d+1))
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
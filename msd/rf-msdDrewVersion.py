import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, recall_score,
                             precision_score, make_scorer)

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

    def load_data(self, genre1=None, genre2=None):
        '''
            Load csv file into X, y, TID -- removes any examples with missing 
            features

        '''
        # grab songs from csv files, remove any with missing features
        msd_csv_path = 'features_vs_genre.csv'
        df = pd.read_csv(msd_csv_path)
        df = df[df.year != 0].dropna(axis=0, how='any')

        if(genre1 != None):
            # Only grab examples of specified genres
            df = df.loc[df['genre'].isin([genre1, genre2])]
            print(df)
            
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
    scaler = StandardScaler()
    data = MSDData()
    #data.load_data()
    data.load_data('Blues', 'Rap')

    X = data.X
    y = data.y
    n, d = X.shape

    index_array = np.arange(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=data.y,
                                         train_size=0.9, random_state=123)

    # select examples from split
    #X_train = data.X[sample_indices]
    #y_train = data.y[sample_indices]
    print(np.unique(y_train, return_counts=True))


    parameters = {
        'clf__max_features': tuple(np.arange(1, d+1)),
        'clf__n_estimators': np.arange(1, 11),
        'clf__max_depth': np.arange(1, 10)
        # 'clf__strategy': ('stratified', 'most_frequent', 'prior', 'uniform')
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(class_weight="balanced"))
    ])

    # grid search with respect to different metrics and print results
    scoring = ['accuracy']#'f1', 'precision', 'recall', 'roc_auc']
    f1 = make_scorer(f1_score, average='weighted')
    scoring = [f1]
    for metric in scoring:
        grid_search = GridSearchCV(
            pipeline,
            parameters,
            scoring=metric,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        # print params for best fit
        print("Best {}: {}".format(metric, grid_search.best_score_))
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        #print(grid_search.cv_results_['mean_train_score'])

        X_train_standardized = scaler.fit_transform(X_train)
        X_test_standardized = scaler.fit_transform(X_test)

        y_predict = grid_search.best_estimator_.predict(X_train_standardized)
        print(metrics.f1_score(y_train, y_predict, average="weighted"))
        #print(metrics.accuracy_score(y_train, y_predict))

        y_predict_test = grid_search.best_estimator_.predict(X_test_standardized)
        print(metrics.f1_score(y_test, y_predict_test, average="weighted"))
        #print(metrics.accuracy_score(y_test, y_predict_test))

        plot_confusion_matrix(metrics.confusion_matrix(y_test, y_predict_test), list(set(y_train)))



def bestFeatures():
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    sfm = SelectFromModel(clf, threshold=0.15)
    sfm.fit(X_train, y_train)

    feat_labels = ["year", "key", "mode", "time_signature", "duration", "end_of_fade_in", "loudness", "song_hotttnesss", "tempo"]
    for feature_list_index in sfm.get_support(indices=True):
        print(feat_labels[feature_list_index])



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.show()


if __name__ == '__main__':
    main()
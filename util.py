import os
import csv

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class MSDData:
    '''
    Class for storing msd data.
    '''
    def __init__(self, X=None, y=None, TID=None, colnames=None):
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
        self.colnames = colnames

    def load_data(self, genre1=None, genre2=None):
        '''
            Load csv file into X, y, TID -- removes any examples with missing
            features

        '''
        # grab songs from csv files, remove any with missing year or any feature with nan
        msd_csv_path = 'data/msd/msd_joined.csv'
        df = pd.read_csv(msd_csv_path)
        # drop year and song hotttnesss
        # df = df.drop(['year', 'song_hotttnesss'], axis=1)
        # df = df[df.year != 0].dropna(axis=0, how='any')

        colnames = list(df.columns)
        colnames.remove('song_id')
        colnames.remove('genre')

        self.colnames = np.array(colnames)

        if(genre1 != None):
            # Only grab examples of specified genres
            df = df.loc[df['genre'].isin([genre1, genre2])]
            print(df)
        TID = df.as_matrix(columns=[df.columns[0]]).flatten()
        X = df.as_matrix(columns=df.columns[2:])
        y = df.as_matrix(columns=[df.columns[1]]).flatten()

        # convert genre labels to numerical representation
        le = LabelEncoder()
        le.fit(y.ravel())
        y = le.transform(y.ravel())

        self.X = X
        self.y = y
        self.TID = TID

        return self


class MusixMatchData:
    '''
    Class for storing musixmatch data.
    '''
    def __init__(self, X=None, y=None, TID=None, MXMID=None, vocab=None):
        '''
        Data class.

        Attributes
        --------------------
        X       -- numpy array of shape (n,d), features
        y       -- numpy array of shape (n,1), labels
        TID     -- numpy array of shape (n,1), track IDs
        MXMID   -- numpy array of shape (n,1), musixmatch IDs
        vocab   -- numpy array of shape (d,1), vocabulary (BOW names)


        '''
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
        self.TID = TID
        self.MXMID = MXMID
        self.vocab = vocab
        if y is not None:
            label_encoder = LabelEncoder()
            label_encoder.fit(y.ravel())
            self.label_encoder = label_encoder
        else:
            self.label_encoder = None

    def write_to_pickle(self, X_filename, genre_filename, pickled_data_path,
                        suffix):
        '''
        Load text files into X, TID, and MXMID and store as pickled objects.

        Parameters
        --------------------
            X_filename          -- str, filename of predictors
            genre_filename      -- str, filename of genre file
            pickled_data_path   -- str, directory path at which to store files
            suffix              -- str, suffix to add to end of files before
                                   writing to pickled_data_path
        '''
        # define dimensions (hard coded for now)
        # n = 73658 # num observations in genres.csv INT mxm_dataset_train
        n = 191402 # num observations in genres.csv
        d = 5000 # number of words in training data vocabulary

        # load genres (the response) and associated track ID
        y = np.empty(n, dtype=object)
        genre_TID = np.empty(n, dtype=object)

        with open(genre_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                # progress update
                if i % 1000 == 0:
                    print(i)

                # store genre and track id
                genre_TID[i] = row[0]
                y[i] = row[1]

        # set for faster lookup time
        tracks_to_keep = set(genre_TID)

        # load predictors
        X = np.zeros((n, d))
        MXMID = np.empty(n, dtype=object)
        # store other TID for joining purposes
        mxm_TID = np.empty(n, dtype=object)
        with open(X_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            num_rows_skipped = 0
            for i, row in enumerate(reader):
                # progress update
                if i % 1000 == 0:
                    print(i)

                # keep track of row indices for observations
                row_index = i - num_rows_skipped

                if len(row) == 0:
                    num_rows_skipped += 1
                    continue

                if row[0][0] == '%':
                    # store vocabulary as array
                    vocab = np.array([row[0][1:]] + row[1:])

                if row[0][0] in ['#', '%']:
                    # update skipped rows count
                    num_rows_skipped += 1
                    continue

                if row[0] not in tracks_to_keep:
                    # skip track if we don't have a genre label for it
                    num_rows_skipped += 1
                    continue

                # store MXMID and TID
                mxm_TID[row_index] = row[0]
                MXMID[row_index] = row[1]

                for item in row[2:]:
                    # each item is 'index:count' i.e. '4:2'
                    item = item.split(':')
                    # offset by two for TID and MXMID
                    col_index = int(item[0]) - 1
                    # store word count as int
                    word_count = int(item[-1])
                    X[row_index, col_index] = word_count

        # make X sparse matrix to save space
        # X = csr_matrix(X)
        # mmwrite(pickled_data_path+'X', X)

        # drop extra space for TIDs that are in genre data but not in mxm data
        # these show up as None in mxm_TID array due to initialization
        rows_to_keep = np.argwhere(mxm_TID != None).flatten()
        mxm_TID = mxm_TID[rows_to_keep]
        X = X[rows_to_keep]
        MXMID = MXMID[rows_to_keep]

        # drop labels for TIDs that are in genre data but not in mxm data
        mxm_TID_set = set(str(s) for s in mxm_TID)
        rows_to_keep = np.array([id in mxm_TID_set for id in genre_TID])
        genre_TID = genre_TID[rows_to_keep]
        y = y[rows_to_keep]

        # sort genre data by TID
        genre_sort_indices = np.argsort(genre_TID)
        TID = genre_TID[genre_sort_indices]
        y = y[genre_sort_indices]

        # sort mxm data by TID
        mxm_sort_indices = np.argsort(mxm_TID)
        X = X[mxm_sort_indices]
        MXMID = MXMID[mxm_sort_indices]

        # set as member data
        self.X = X
        self.y = y
        self.TID = TID
        self.MXMID = MXMID
        self.vocab = vocab

        # set label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(y.ravel())
        self.label_encoder = label_encoder

        # store as pickle objects
        print('Writing data to disk at: {}'.format(pickled_data_path))
        ensure_directory(pickled_data_path)
        np.save(pickled_data_path+'TID'+'_'+suffix, TID)
        np.save(pickled_data_path+'MXMID'+'_'+suffix, MXMID)
        np.save(pickled_data_path+'vocab'+'_'+suffix, vocab)
        np.save(pickled_data_path+'X'+'_'+suffix, X)
        np.save(pickled_data_path+'y'+'_'+suffix, y)

        return self


    def load_from_pickle(self, pickled_data_path, suffix):
        '''
        Load pickled files into X, y, TID, MXMID, and vocab.
        '''
        print('Loading data from: {}'.format(pickled_data_path))
        self.TID = np.load(pickled_data_path+'TID'+'_'+suffix+'.npy')
        self.MXMID = np.load(pickled_data_path+'MXMID'+'_'+suffix+'.npy')
        self.X = np.load(pickled_data_path+'X'+'_'+suffix+'.npy')
        self.y = np.load(pickled_data_path+'y'+'_'+suffix+'.npy')
        # self.X = mmread(pickled_data_path + 'X.mtx')
        self.vocab = np.load(pickled_data_path+'vocab'+'_'+suffix+'.npy')

        # set label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(self.y.ravel())
        self.label_encoder = label_encoder

        return self

    def get_one_genre(self, genre_name):
        '''
        Select data specific to genre with name genre_name.

        Parameters
        --------------------
            genre_name  -- str, name of genre to select

        Returns
        --------------------
            X           -- numpy array, observations for given genre
        '''
        X = self.X[self.y == genre_name]
        return X

    def encode_labels(self):
        '''Encode labels using sklearn.LabelEncoder'''
        self.y = self.label_encoder.transform(self.y.ravel())
        return self

    def decode_labels(self):
        '''Decode labels using sklearn.LabelEncoder'''
        self.y = self.label_encoder.inverse_transform(self.y.ravel())
        return self


class MSDMXMData:
    '''Class for joining MSD and MXM data.'''
    def __init__(self, X_train=None, y_train=None, TID_train=None,
                 X_test=None, y_test=None, TID_test=None,
                 colnames=None, label_encoder=None):
        self.X_train = X_train
        self.y_train = y_train
        self.TID_train = TID_train
        self.X_test = X_test
        self.y_test = y_test
        self.TID_test = TID_test
        self.colnames = colnames
        self.label_encoder = label_encoder


    def load_data(self):
        # load MSD
        msd_data = MSDData()
        msd_data.load_data()

        # load MXM
        mxm_data = MusixMatchData()
        try:
            mxm_data.load_from_pickle(
                pickled_data_path='data/musixmatch/pickled/',
                suffix='all')
        except:
            mxm_data.write_to_pickle(
                X_filename='data/musixmatch/mxm_all_data.txt',
                genre_filename='genres.csv',
                pickled_data_path='data/musixmatch/pickled/',
                suffix='all')


        # join data
        assert(set(msd_data.TID.flatten()) == set(mxm_data.TID.flatten()))

        # sort MSD by TID
        sort_indices = np.argsort(msd_data.TID).flatten()
        X_msd = msd_data.X[sort_indices]
        y_msd = msd_data.y[sort_indices]
        colnames_msd = msd_data.colnames

        # sort MXM by TID
        sort_indices = np.argsort(mxm_data.TID)
        X_mxm = mxm_data.X[sort_indices]
        y_mxm = mxm_data.y[sort_indices]
        colnames_mxm = mxm_data.vocab

        assert(np.all(y_msd == mxm_data.label_encoder.transform(y_mxm)))

        # join data
        TID = mxm_data.TID[sort_indices]
        X = np.column_stack((X_mxm, X_msd))
        y = y_mxm

        # train stratified test split
        train_indices, test_indices = train_test_split(
            range(len(TID)),
            stratify=y,
            train_size=0.8,
            random_state=123)

        self.X_train = X[train_indices]
        self.y_train = y[train_indices]
        self.TID_train = TID[train_indices]

        self.X_test = X[test_indices]
        self.y_test = y[test_indices]
        self.TID_test = TID[test_indices]

        self.colnames = np.concatenate((colnames_mxm, colnames_msd))
        self.label_encoder = mxm_data.label_encoder

        # print('X shape: {}'.format(X.shape))
        # print('y shape: {}'.format(y.shape))
        # print('colnames shape: {}'.format(colnames.shape))
        # print('TID shape: {}'.format(TID.shape))

    def encode_labels(self):
        self.y_train = self.label_encoder.transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)
        return self




def ensure_directory(dir_path):
    '''Ensure directory exists at dir_path.'''
    dir = os.path.dirname(__file__)
    dir = os.path.join(dir,dir_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return True


def stratified_random_sample_mxm(data, sample_proportion, random_state):
    '''
    Create stratified random sample of sample_proportion of the data points.

    Parameters
    --------------------
        data                -- MusixMatchData object to sample from
        sample_proportion   -- float, proportion to sample
        random_state        -- int, random seed

    Returns
    --------------------
        sampled_data        -- MusixMatchData object for sampled
                               observations
    '''
    n, _ = data.X.shape
    index_array = np.arange(n)
    sample_indices, _ = train_test_split(
        index_array,
        stratify=data.y,
        train_size=sample_proportion,
        random_state=random_state)

    # select examples from split
    X = data.X[sample_indices]
    y = data.y[sample_indices]
    TID = data.TID[sample_indices]
    MXMID = data.TID[sample_indices]

    sampled_data = MusixMatchData(
        X=X,
        y=y,
        TID=TID,
        MXMID=MXMID,
        vocab=data.vocab)

    return sampled_data


def stratified_random_sample_MXMMSD(data, sample_proportion, random_state):
    '''
    Create stratified random sample of sample_proportion of the data points.

    Parameters
    --------------------
        data                -- MusixMatchData object to sample from
        sample_proportion   -- float, proportion to sample
        random_state        -- int, random seed

    Returns
    --------------------
        sampled_data        -- MusixMatchData object for sampled
                               observations
    '''
    n, _ = data.X_train.shape
    index_array = np.arange(n)
    train_sample_indices, _ = train_test_split(
        index_array,
        stratify=data.y_train,
        train_size=sample_proportion,
        random_state=random_state)

    # select examples from split
    X_train = data.X_train[train_sample_indices]
    y_train = data.y_train[train_sample_indices]
    TID_train = data.TID_train[train_sample_indices]

    sampled_data = MSDMXMData(
        X_train=X_train,
        y_train=y_train,
        X_test=data.X_test,
        y_test=data.y_test,
        TID_train=TID_train,
        TID_test=data.TID_test,
        colnames=data.colnames,
        label_encoder=data.label_encoder
    )

    return sampled_data



def select_genres_mxm(data, genre_list):
    '''
    Select from data all points with label in genre_list.


    Parameters
    --------------------
        data        -- MusixMatchData object to sample from
        genre_list  -- list of str, genres to select

    Returns
    --------------------
        sampled_data        -- MusixMatchData object for sampled
                               observations
    '''
    # get indices to sample
    sample_indices = [index for index,genre in enumerate(data.y) if genre in genre_list]

    # select examples from split
    X = data.X[sample_indices]
    y = data.y[sample_indices]
    TID = data.TID[sample_indices]
    MXMID = data.TID[sample_indices]

    sampled_data = MusixMatchData(
        X=X,
        y=y,
        TID=TID,
        MXMID=MXMID,
        vocab=data.vocab)

    return sampled_data


def select_genres_MXMMSD(data, genre_list):
    '''
    Select from data all points with label in genre_list.


    Parameters
    --------------------
        data        -- MusixMatchData object to sample from
        genre_list  -- list of str, genres to select

    Returns
    --------------------
        sampled_data        -- MusixMatchData object for sampled
                               observations
    '''
    # get indices to sample
    train_sample_indices = [index for index,genre in enumerate(data.y_train) 
                            if genre in genre_list]
    test_sample_indices = [index for index, genre in enumerate(data.y_test) 
                           if genre in genre_list]

    # select examples from split
    X_train = data.X_train[train_sample_indices]
    y_train = data.y_train[train_sample_indices]
    TID_train = data.TID_train[train_sample_indices]

    X_test = data.X_test[test_sample_indices]
    y_test = data.y_test[test_sample_indices]
    TID_test = data.TID_test[test_sample_indices]

    sampled_data = MSDMXMData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        TID_train=TID_train,
        TID_test=TID_test,
        colnames=data.colnames,
        label_encoder=data.label_encoder
    )

    return sampled_data
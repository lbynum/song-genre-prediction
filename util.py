import csv

import numpy as np


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


    def write_to_pickle(self, X_filename, genre_filename):
        '''
        Load text files into X, TID, and MXMID and store as pickled objects.

        Parameters
        --------------------
            X_filename      -- string, filename of predictors
            genre_filename  -- string, filename of genre file
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

        # store as pickle objects
        pickled_data_path = 'data/musixmatch/pickled/'
        print('Writing data to disk at: {}'.format(pickled_data_path))
        np.save(pickled_data_path+'TID', TID)
        np.save(pickled_data_path+'MXMID', MXMID)
        np.save(pickled_data_path+'vocab', vocab)
        np.save(pickled_data_path+'X', X)
        np.save(pickled_data_path+'y', y)


    def load_from_pickle(self):
        '''
        Load pickled files into X, y, TID, MXMID, and vocab.
        '''
        pickled_data_path = 'data/musixmatch/pickled/'
        print('Loading data from: {}'.format(pickled_data_path))
        self.TID = np.load(pickled_data_path+'TID.npy')
        self.MXMID = np.load(pickled_data_path+'MXMID.npy')
        self.X = np.load(pickled_data_path+'X.npy')
        self.y = np.load(pickled_data_path+'y.npy')
        # self.X = mmread(pickled_data_path + 'X.mtx')
        self.vocab = np.load(pickled_data_path+'vocab.npy')


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

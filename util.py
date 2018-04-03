import csv
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmwrite, mmread

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

    def write_to_pickle(self, filename, n, d):
        '''
        Load text files into X, TID, and MXMID and store as pickled objects.

        Parameters
        --------------------
            filename    -- string, filename
            n           -- int, number of observations
            d           -- int, number of features
        '''
        X = np.zeros((n, d))
        TID = np.empty(n, dtype=object)
        MXMID = np.empty(n, dtype=object)
        with open(filename, 'r') as csvfile:
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

                # store TID and MXMID
                TID[row_index] = row[0]
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

        # set as member data
        self.X = X
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

    def load_from_pickle(self):
        '''
        Load pickled files into X, TID, and MXMID.

        Parameters
        --------------------
            filename    -- string, filename
            n           -- int, number of observations
            d           -- int, number of features
        '''
        pickled_data_path = 'data/musixmatch/pickled/'
        print('Loading data from: {}'.format(pickled_data_path))
        self.TID = np.load(pickled_data_path+'TID.npy')
        self.MXMID = np.load(pickled_data_path+'MXMID.npy')
        self.X = np.load(pickled_data_path+'X.npy')
        # self.X = mmread(pickled_data_path + 'X.mtx')
        self.vocab = np.load(pickled_data_path+'vocab.npy')



# mmdata = MusixMatchData()
# mmdata.write_to_pickle(filename='data/musixmatch/mxm_dataset_train.txt',
#                        n=210519,
#                        d=5000)
# mmdata.load_from_pickle()

import csv
import pickle

import numpy as np

try:
    pickled_data_path = 'data/musixmatch/pickled/'
    TID_list_train = pickle.load(open(pickled_data_path + 'TID_list_train', 'rb'))
    MXMID_list_train = pickle.load(open(pickled_data_path + 'MXMID_list_train', 'rb'))
    X_train = np.load(pickled_data_path + 'X_train.npy')

    print 'X_train: {} array'.format(X_train.shape)
    print 'TID_list_train: list of {}'.format(len(TID_list_train))
    print 'MXMID_list_train: list of {}'.format(len(MXMID_list_train))

    print '''\nFiles already created -- try the following to load data:

    pickled_data_path = 'data/musixmatch/pickled/'
    TID_list_train = pickle.load(open(pickled_data_path+'TID_list_train', 'rb'))
    MXMID_list_train = pickle.load(open(pickled_data_path+'MXMID_list_train', 'rb'))
    X_train = np.load(pickled_data_path+'X_train.npy')
    '''

except:
    raw_data_path = 'data/musixmatch/mxm_dataset_train.txt'
    num_observations = 210519
    X_train = np.empty((210519, 5000))

    TID_list = []
    MXMID_list = []
    with open(raw_data_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        num_rows_skipped = 0
        for i, row in enumerate(reader):
            # progress update
            if i % 1000 == 0:
                print i

            # keep track of row indices for observations
            row_index = i - num_rows_skipped
            if row[0][0] == '%':
                col_names = ['TID', 'MXMID'] + [row[0][1:]] + row[1:]

            if row[0][0] in ['#', '%']:
                # update skipped rows count
                num_rows_skipped += 1
                continue

            # store TID and MXMID
            TID_list.append(row[0])
            MXMID_list.append(row[1])

            for item in row[2:]:
                # each item is 'index:count' i.e. '4:2'
                item = item.split(':')
                # offset by two for TID and MXMID
                col_index = int(item[0]) - 1
                # store word count as int
                word_count = int(item[-1])
                X_train[row_index, col_index] = word_count

    # store as pickle objects
    pickle.dump(TID_list, open(pickled_data_path+'TID_list_train', 'wb'))
    pickle.dump(MXMID_list, open(pickled_data_path+'MXMID_list_train', 'wb'))
    np.save(pickled_data_path+'X_train', X_train)

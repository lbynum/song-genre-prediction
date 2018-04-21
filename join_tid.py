import csv

import pandas as pd


# get all musixmatch TIDs for train and test data
train_mxm_TIDs = []
with open('data/musixmatch/mxm_dataset_train.txt', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        # progress update
        if i % 10000 == 0:
            print(i)
        if i > 17:
            train_mxm_TIDs.append(row[0])
test_mxm_TIDs = []
with open('data/musixmatch/mxm_dataset_test.txt', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        # progress update
        if i % 10000 == 0:
            print(i)
        if i > 17:
            test_mxm_TIDs.append(row[0])

# mxm_ids = set(mxm_TIDs)
# mxm_df = pd.read_csv('musixmatch_tid.csv')
# mxm_ids = set(mxm_df.iloc[0:,0])

# get all MSD TIDs
msd_df = pd.read_csv('msd_tids.csv')
msd_ids = set(msd_df.iloc[0:,0])

# get all original genre lables
genres = pd.read_csv('genres.csv')

# find intersection among training data, join with genre labels, and save as csv
train_mxm_TIDs = set(train_mxm_TIDs)
intersection_train = train_mxm_TIDs.intersection(msd_ids)
train_genres = genres[[TID in intersection_train for TID in genres.iloc[:,0]]]
train_genres.to_csv('joined_genres_train.csv', index=False)

# find intersection among test data, join with genre labels, and save as csv
test_mxm_TIDs = set(test_mxm_TIDs)
intersection_test = test_mxm_TIDs.intersection(msd_ids)
test_genres = genres[[TID in intersection_test for TID in genres.iloc[:,0]]]
test_genres.to_csv('joined_genres_test.csv', index=False)

all_TIDs = intersection_test.union(intersection_train)
all_genres = genres[[TID in all_TIDs for TID in genres.iloc[:,0]]]
all_genres.to_csv('joined_genres.csv', index=False)



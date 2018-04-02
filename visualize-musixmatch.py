import pickle

import numpy as np



def load_data():
    pickled_data_path = 'data/musixmatch/pickled/'
    TID_list_train = pickle.load(
        open(pickled_data_path + 'TID_list_train', 'rb'))
    MXMID_list_train = pickle.load(
        open(pickled_data_path + 'MXMID_list_train', 'rb'))
    X_train = np.load(pickled_data_path + 'X_train.npy')



def main():
    pass






if __name__ == '__main__':
    main()
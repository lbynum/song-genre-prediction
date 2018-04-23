import matplotlib.pyplot as plt
import pandas as pd

from util import MSDMXMData, select_genres_MXMMSD


def main():
    data = MSDMXMData()
    data.load_data()

    genre_list = ['Pop', 'Rap']
    data = select_genres_MXMMSD(data, genre_list=genre_list)

    X_train = data.X_train[:,-10:]
    colnames = data.colnames[-10:]

    df = pd.DataFrame(X_train, columns=colnames)
    df['genre'] = data.y_train
    df.to_csv('msd_train.csv')

    # n, d = X_train.shape
    # plt.boxplot(X_train)
    # plt.xticks(list(range(1,d+1)), colnames)
    # plt.show()

if __name__ == '__main__':
    main()
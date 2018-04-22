import matplotlib.pyplot as plt

from util import MSDMXMData, select_genres_MXMMSD


def main():
    data = MSDMXMData()
    data.load_data()

    genre_list = ['Pop', 'Rap']
    data = select_genres_MXMMSD(data, genre_list=genre_list)

    X_train = data.X_train[:,-10:]
    colnames = data.colnames[-10:]

    n, d = X_train.shape
    plt.boxplot(X_train)
    plt.xticks(list(range(1,d+1)), colnames)
    plt.show()

if __name__ == '__main__':
    main()
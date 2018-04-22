from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

from util import MSDMXMData, select_genres_MXMMSD

def barplot_top_n(word_counts, n, y_label='', title=''):
    # sort by count
    sorted_pairs = sorted(word_counts, key=lambda pair: pair[1], reverse=True)

    # select words and counts for plot
    words = [pair[0] for pair in sorted_pairs[:n]]
    counts = [pair[1] for pair in sorted_pairs[:n]]

    plt.figure()
    y_pos = np.arange(len(words))
    plt.bar(y_pos, counts)
    plt.xticks(y_pos, words)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def remove_stopwords(word_counts):
    stop_words = set(stopwords.words('english'))
    word_counts = [pair for pair in word_counts
                   if pair[0] not in stop_words]
    return word_counts

def compute_total_occurrences(X, vocab):
    ## compute total word counts
    print('Computing total word counts...')
    col_sums = np.sum(X, axis=0)

    # add words to counts (make list so zip object playes nice)
    word_counts = list(zip(vocab, col_sums))

    return word_counts

def compute_song_occurrences(X, vocab):
    print('Computing word counts across songs...')
    # compute word indicators
    indicators = X > 0
    # sum across indicators to count number of songs each word shows up in
    col_sums = np.sum(indicators, axis=0)
    # add words to counts
    word_counts = list(zip(vocab, col_sums))

    return word_counts


def main():
    # load musixmatch data
    # data = MusixMatchData()
    # try:
    #     data.load_from_pickle(pickled_data_path='data/musixmatch/pickled/',
    #                           suffix='train')
    # except:
    #     data.write_to_pickle(
    #         X_filename='data/musixmatch/mxm_dataset_train.txt',
    #         genre_filename='genres.csv',
    #         pickled_data_path='data/musixmatch/pickled/',
    #         suffix='train'
    #     )
    data = MSDMXMData()
    data.load_data()
    X_train = data.X_train[:,:-10]
    vocab = data.colnames[:-10]

    # word_counts = compute_total_occurrences(X_train, vocab)
    #
    # ## barplot for highest occurring words
    # barplot_top_n(word_counts,
    #               n=20,
    #               y_label='Occurrences in corpus',
    #               title='Highest occurring words -- All genres')
    #
    # ## barplot for highest occurring words without stopwords
    # barplot_top_n(remove_stopwords(word_counts),
    #               n=20,
    #               y_label='Occurrences in corpus',
    #               title='Highest occurring words without stopwords -- All genres')
    #
    # word_counts = compute_song_occurrences(X_train, vocab)
    #
    # ## bar plot -- most popular words across all songs
    # barplot_top_n(word_counts,
    #               n=20,
    #               y_label='Number of songs',
    #               title='Most popular words -- All genres')
    #
    # ## bar plot -- most popular words across all songs without stopwords
    # barplot_top_n(remove_stopwords(word_counts),
    #               n=20,
    #               y_label='Number of songs',
    #               title='Most popular words without stopwords -- All genres')


    # genres = np.unique(data.y)
    genres = ['Rap', 'Pop']
    for genre in genres:
        data_sample = select_genres_MXMMSD(data, [genre])
        X_train = data_sample.X_train[:,:-10]
        # most occurring
        word_counts = compute_total_occurrences(X_train, vocab)
        barplot_top_n(remove_stopwords(word_counts),
                      n=20,
                      y_label='Occurrences in corpus',
                      title='Highest occurring words without stopwords -- {}'.format(genre))
        # most popular
        word_counts = compute_song_occurrences(X_train, vocab)
        barplot_top_n(remove_stopwords(word_counts),
                      n=20,
                      y_label='Number of songs',
                      title='Most popular words without stopwords -- {}'.format(
                          genre))


if __name__ == '__main__':
    main()
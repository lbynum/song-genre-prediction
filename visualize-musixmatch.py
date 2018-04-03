from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

from util import MusixMatchData

def barplot_top_n(word_counts, n, y_label='', title=''):
    # sort by count
    word_counts = sorted(word_counts, key=lambda pair: pair[1], reverse=True)

    # select words and counts for plot
    words = [pair[0] for pair in word_counts[:n]]
    counts = [pair[1] for pair in word_counts[:n]]

    plt.figure()
    y_pos = np.arange(len(words))
    plt.bar(y_pos, counts)
    plt.xticks(y_pos, words)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()



def main():
    # load musixmatch data
    data = MusixMatchData()
    try:
        data.load_from_pickle()
    except:
        data.write_to_pickle(
            X_filename='data/musixmatch/mxm_dataset_train.txt',
            genre_filename='genres.csv')

    ## compute total word counts
    print('Computing total word counts...')
    col_sums = np.sum(data.X, axis=0)
    # add words to counts
    word_counts = zip(data.vocab, col_sums)

    ## barplot for highest occurring words
    barplot_top_n(word_counts,
                  n=20,
                  y_label='Number of occurrences in corpus',
                  title='Highest occurring words')

    ## barplot for highest occurring words minus stopwords
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    print(word_counts)
    word_counts = [pair for pair in word_counts
                   if pair[0] not in stop_words]
    print(word_counts)
    barplot_top_n(word_counts,
                  n=20,
                  y_label='Number of occurrences in corpus',
                  title='Highest occurring words without stopwords')


    ## bar plot -- most popular words across all songs
    print('Computing word counts across songs...')
    # compute word indicators
    indicators = data.X > 0
    # sum across indicators to count number of songs each word shows up in
    col_sums = np.sum(indicators, axis=0)
    # add words to counts
    word_counts = zip(data.vocab, col_sums)

    barplot_top_n(word_counts,
                  n=20,
                  y_label='Number of songs',
                  title='Most popular words')

    ## bar plot -- most popular words across all songs without stopwords
    word_counts = [pair for pair in word_counts
                   if pair[0] not in stop_words]

    barplot_top_n(word_counts,
                  n=20,
                  y_label='Number of songs',
                  title='Most popular words without stopwords')



    # TODO: look at same plot for each genre





if __name__ == '__main__':
    main()
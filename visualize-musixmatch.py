from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

from util import MusixMatchData



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
    # sort by count
    word_counts = sorted(word_counts, key=lambda pair: pair[1], reverse=True)

    ## barplot for highest occuring words
    words = [pair[0] for pair in word_counts[:20]]
    counts = [pair[1] for pair in word_counts[:20]]
    y_pos = np.arange(len(words))
    plt.bar(y_pos, counts)
    plt.xticks(y_pos, words)
    plt.title('Highest occuring words')
    plt.show()

    ## barplot for highest occuring words minus stopwords
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    word_counts = [pair for pair in word_counts
                   if pair[0] not in stop_words]
    words = [pair[0] for pair in word_counts[:20]]
    counts = [pair[1] for pair in word_counts[:20]]
    y_pos = np.arange(len(words))
    plt.bar(y_pos, counts)
    plt.xticks(y_pos, words)
    plt.title('Highest occuring words without stopwords')
    plt.show()


    # TODO: also do most popular i.e. binary instead of counts
    # TODO: look at same plot for each genre





if __name__ == '__main__':
    main()
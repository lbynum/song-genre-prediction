SEVERAL DESCRIPTIONS BELOW TAKEN FROM: https://labrosa.ee.columbia.edu/millionsong/musixmatch#getting

**full_word_list** : The full list of stemmed words and the total word counts, i.e. all the words that were seen at least once. There are 498,134 unique words, for a total of 55,163,335 occurrences. The 5,000 words in the dataset account for 50,607,582 occurrences, so roughly 92%. NOTE 1: for choosing our 5,000 words, we normalized the word counts by the number of word occurrences in each song. Thus, it is not the top 5,000 of this file. NOTE 2: the list is super noisy, we know it! We made sure that the top 5,000 words was clean, but for the rest, no guarantee whatsoever, the bottom of the list is a mess (punctuation signs, foreign symbols, words glued together, ... name it, it's there).

**mxm_779k_matches** : The full list of 779K matches to the Million Song Dataset.

**mxm_dataset_test** :  210,519 training bag-of-words lyrics for many MSD tracks: each track is described as the word-counts for a dictionary of the top 5,000 words across the set.

**mxm_dataset_train** : 27,143 testing bag-of-words
	The two text files are formatted as follow (per line):
	# - comment, ignore
	%word1,word2,... - list of top words, in popularity order
	TID,MXMID,idx:cnt,idx:cnt,... - track ID from MSD, track ID from musiXmatch,
	then word index : word count (word index starts at 1!)

**mxm_reverse_mapping** : list of unstemmed words with their stemmed version (one of many possible mappings because stemming is not one-to-one)

**stemmed_words** : list of 10K popular English words and how they would appear in the dataset to help get a better intuition for how to deal with weird edge cases of stemming (For example, "I'm" is never seen, it becomes "I am". Note the mistake we made with "n't " -> "n not", should have been -> "n not ". It explains why "can't" becomes "ca not".)

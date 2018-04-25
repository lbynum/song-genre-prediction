# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# import pandas
import pandas as pd

# imports to use MSD
import sys
import time
import glob
import datetime
import sqlite3
import numpy as np  # get it at: http://numpy.scipy.org/

# path to the Million Song Dataset subset (uncompressed)
# CHANGE IT TO YOUR LOCAL CONFIGURATION
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname('MillionSongSubset')))
msd_subset_path = __location__ + '/MillionSongSubset'
# print "here is the path: ", msd_subset_path
msd_subset_data_path = os.path.join(msd_subset_path, 'data')
msd_subset_addf_path = os.path.join(msd_subset_path, 'AdditionalFiles')
# assert os.path.isdir(msd_subset_path), 'wrong path'  # sanity check
# path to the Million Song Dataset code
# CHANGE IT TO YOUR LOCAL CONFIGURATION
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname('MSongsDB')))
msd_code_path = __location__
assert os.path.isdir(msd_code_path), 'wrong path'  # sanity check
# we add some paths to python so we can import MSD code
# Ubuntu: you can change the environment variable PYTHONPATH
# in your .bashrc file so you do not have to type these lines
path = os.path.join(msd_code_path, 'MSongsDB', 'PythonSrc')
sys.path.append(path)

# imports specific to the MSD
import hdf5_getters as GETTERS


def get_all_examples(basedir, genre_dict, ext='.h5'):
    """
    From a base directory, goes through all subdirectories,
    and grabs all songs and their features and puts them into a pandas dataframe
    INPUT
       basedir    - base directory of the dataset
       genre_dict - a dictionary mapping track id to genre based tagraum dataset
       ext        - extension, .h5 by default
    RETURN
       dataframe containing all song examples
    """
    features_vs_genre = pd.DataFrame()

    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))
        # # count files
        # count += len(files)
        # apply function to all files
        for f in files:
            h5 = GETTERS.open_h5_file_read(f)
            num_songs = GETTERS.get_num_songs(h5)
            for i in range(num_songs):
                if i % 10000 == 0:
                    print(i)
                song_id = GETTERS.get_track_id(h5, i).decode('utf-8')
                if (song_id in genre_dict):
                    genre = genre_dict[song_id]
                    year = GETTERS.get_year(h5, i)
                    duration = GETTERS.get_duration(h5, i)
                    end_of_fade_in = GETTERS.get_end_of_fade_in(h5, i)
                    loudness = GETTERS.get_loudness(h5, i)
                    song_hotttnesss = GETTERS.get_song_hotttnesss(h5, i)
                    tempo = GETTERS.get_tempo(h5, i)
                    key = GETTERS.get_key(h5, i)
                    key_confidence = GETTERS.get_key_confidence(h5, i)
                    mode = GETTERS.get_mode(h5, i)
                    mode_confidence = GETTERS.get_mode_confidence(h5, i)
                    time_signature = GETTERS.get_time_signature(h5, i)
                    time_signature_confidence = GETTERS.get_time_signature_confidence(
                        h5, i)
                    artist_name = GETTERS.get_artist_name(h5)
                    title = GETTERS.get_title(h5)
                    # length of sections_start array gives us number of start
                    # num_sections = len(GETTERS.get_sections_start(h5))
                    # num_segments = len(GETTERS.get_segments_confidence(h5))
                    example = pd.DataFrame(data=[(artist_name, title, song_id,
                                                  genre, year, key,
                                                  key_confidence, mode,
                                                  mode_confidence,
                                                  time_signature,
                                                  time_signature_confidence,
                                                  duration,
                                                  end_of_fade_in, loudness,
                                                  song_hotttnesss, tempo)],
                                           columns=['artist_name', 'title',
                                                    'song_id', 'genre', 'year',
                                                    'key', 'key_confidence',
                                                    'mode', 'mode_confidence',
                                                    'time_signature',
                                                    'time_signature_confidence',
                                                    'duration',
                                                    'end_of_fade_in',
                                                    'loudness',
                                                    'song_hotttnesss',
                                                    'tempo'])
                    features_vs_genre = features_vs_genre.append(example)
            h5.close()

    return features_vs_genre


def main():
    genres_data_frame = pd.read_csv(msd_code_path + '/../genres.csv',
                                    dtype={'song_id': str, 'genre': str})

    unique_genres = genres_data_frame['genre'].unique()

    # Create the dictionary of songs to genres.
    genre_dict = {}
    for col in range(genres_data_frame.shape[0]):
        genre_dict[genres_data_frame['song_id'][col]] = \
        genres_data_frame['genre'][col]

    # write csv file
    features_vs_genre = get_all_examples('./', genre_dict)
    df1 = features_vs_genre[['song_id', 'genre']]
    features_vs_genre.to_csv('./msd_features.csv', index=False)
    df1.to_csv('./msd_tids.csv')


if _name_ == "_main_":
    main()
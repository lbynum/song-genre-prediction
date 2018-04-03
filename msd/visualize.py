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
import numpy as np # get it at: http://numpy.scipy.org/
# path to the Million Song Dataset subset (uncompressed)
# CHANGE IT TO YOUR LOCAL CONFIGURATION
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname('MillionSongSubset')))
msd_subset_path=__location__ + '/MillionSongSubset'
# print "here is the path: ", msd_subset_path
msd_subset_data_path=os.path.join(msd_subset_path,'data')
msd_subset_addf_path=os.path.join(msd_subset_path,'AdditionalFiles')
assert os.path.isdir(msd_subset_path),'wrong path' # sanity check
# path to the Million Song Dataset code
# CHANGE IT TO YOUR LOCAL CONFIGURATION
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname('MSongsDB')))
msd_code_path=__location__
assert os.path.isdir(msd_code_path),'wrong path' # sanity check
# we add some paths to python so we can import MSD code
# Ubuntu: you can change the environment variable PYTHONPATH
# in your .bashrc file so you do not have to type these lines
path = os.path.join(msd_code_path,'MSongsDB', 'PythonSrc')
sys.path.append( path )

# imports specific to the MSD
import hdf5_getters as GETTERS

# we define this very useful function to iterate the files
def avg_feature_all_files(basedir, genre_dict, ext='.h5'):
    """
    From a base directory, go through all subdirectories,
    find all files with the given extension, apply the
    given function 'func' to all of them.
    If no 'func' is passed, we do nothing except counting.
    INPUT
       basedir  - base directory of the dataset
       func     - function to apply to all filenames
       ext      - extension, .h5 by default
    RETURN
       number of files
    """
    features_vs_genre = pd.DataFrame()

    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        # # count files
        # count += len(files)
        # apply function to all files
        for f in files :
            h5 = GETTERS.open_h5_file_read(f)
            song_id = GETTERS.get_track_id(h5).decode('utf-8')
            if (song_id in genre_dict):
                genre = genre_dict[song_id]
                danceability = GETTERS.get_danceability(h5)
                duration = GETTERS.get_duration(h5)
                end_of_fade_in = GETTERS.get_end_of_fade_in(h5)
                energy = GETTERS.get_energy(h5)
                loudness = GETTERS.get_loudness(h5)
                song_hotttnesss = GETTERS.get_song_hotttnesss(h5)
                tempo = GETTERS.get_tempo(h5)
                example = pd.DataFrame(data=[(genre, danceability, duration, 
                                              end_of_fade_in, energy, loudness, 
                                              song_hotttnesss, tempo)], 
                                       columns=['genre', 'danceability', 
                                                'duration', 'end_of_fade_in', 
                                                'energy', 'loudness', 
                                                'song_hotttnesss', 'tempo'])
                features_vs_genre = features_vs_genre.append(example)
            h5.close()

    return genre_vs_features

def main():
    genres_data_frame = pd.read_csv(msd_code_path+'/../genres.csv', 
                                    dtype={'song_id': str, 'genre': str})

    unique_genres = genres_data_frame['genre'].unique()

    # Create the dictionary of songs to genres.
    genre_dict = {}
    for col in range(genres_data_frame.shape[0]):
        genre_dict[genres_data_frame['song_id'][col]] = genres_data_frame['genre'][col]
        # print(genres_data_frame['song_id'][col], genres_data_frame['genre'][col])

    features_vs_genre = avg_feature_all_files(msd_subset_data_path, genre_dict)
    
    # write csv file
    features_vs_genre.to_csv('./features_vs_genre.csv')

        

if __name__ == "__main__":
    main()
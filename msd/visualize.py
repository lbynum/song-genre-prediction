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

all_artist_names = set()
# we define this very useful function to iterate the files
def apply_to_all_files(basedir,func=lambda x: x,ext='.h5'):
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
    count = 0
    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        # count files
        count += len(files)
        # apply function to all files
        for f in files :
            func(f)       
    return count

def extract_features(filename):
    h5 = GETTERS.open_h5_file_read(filename)
    artist_name = GETTERS.get_artist_name(h5)
    print(artist_name)
    all_artist_names.add( artist_name )
    h5.close()

def main():
    genres_data_frame = pd.read_csv(msd_code_path+'/../genres.csv', 
                                    dtype={'song_id': str, 'genre': str})

    apply_to_all_files(msd_subset_data_path, func=extract_features)
    print(len(all_artist_names))





if __name__ == "__main__":
    main()
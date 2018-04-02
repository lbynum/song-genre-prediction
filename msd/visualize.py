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
msd_subset_path='/home/thierry/Desktop/MillionSongSubset'
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
msd_code_path=__location__#'/home/thierry/Columbia/MSongsDB'
# print msd_code_path + '/MSongsDB'
assert os.path.isdir(msd_code_path),'wrong path' # sanity check
# we add some paths to python so we can import MSD code
# Ubuntu: you can change the environment variable PYTHONPATH
# in your .bashrc file so you do not have to type these lines
path = os.path.join(msd_code_path,'MSongsDB', 'PythonSrc')
# print path
sys.path.append( path )

# imports specific to the MSD
import hdf5_getters as GETTERS

def import_genres(csv_filepath):
	data_frame = pd.read_csv(csv_filepath,)
	# print(data_frame)
	print(data_frame.loc[data_frame['song_id'] == 'TRAAGEC128E079252C', 'genre'])

def main():
	import_genres(msd_code_path+'/../genres.csv')

if __name__ == "__main__":
    main()
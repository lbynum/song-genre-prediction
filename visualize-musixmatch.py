import pickle

import numpy as np
import pandas as pd

from util import MusixMatchData



def main():
    # load musixmatch data
    data = MusixMatchData()
    try:
        data.load_from_pickle()
    except:
        data.write_to_pickle()










if __name__ == '__main__':
    main()
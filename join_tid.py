import pandas as pd

mxm_df = pd.read_csv('musixmatch_tid.csv')
mxm_ids = set(mxm_df.iloc[0:,0])

msd_df = pd.read_csv('msd_tids.csv')
msd_ids = set(msd_df.iloc[0:,1])

intersection = mxm_ids.intersection(msd_ids)

intersection_df = msd_df[[TID in intersection for TID in list(msd_df.iloc[:,0])]]

intersection_df.to_csv('joined_tid.csv')



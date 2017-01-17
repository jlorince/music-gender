import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,cosine
import time
from os.path import basename
import pickle
import datetime
import multiprocessing as mp
import empty_module
from tqdm import tqdm as tq

### FILTERS
filter_gender = ['m','f']
#filter_playcount = 1000

### SUPPORT FUNCTIONS

def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])


def songs_per_artist(uid):
    df = parse_df('P:/Projects/BigMusic/jared.IU/scrobbles-complete/{}.txt'.format(uid))
    return uid,len(df)/float(len(df.artist_id.unique()))


if __name__ == '__main__':

    import sys
    from glob import glob
    import math
    import time,datetime
    import os
    import itertools

    n_procs = mp.cpu_count()
  

    ### METADATA HANDLING
    user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'])

    user_data['sample_playcount'][user_data['sample_playcount']=='\\N'] = 0 
    user_data['sample_playcount'] = user_data['sample_playcount'].astype(int)

    filtered = user_data.loc[(user_data['gender'].isin(filter_gender)) & (user_data['sample_playcount']>0)][['user_id','gender','sample_playcount']]

    ids_f = set(filtered[filtered['gender']=='f']['user_id'])
    ids_m = set(filtered[filtered['gender']=='m']['user_id'])


    all_files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')

    #files_m = sorted([f for f in all_files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_m],key=os.path.getsize,reverse=True)
    #files_f = sorted([f for f in all_files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_f],key=os.path.getsize,reverse=True)

    ### RUN MAIN PROCESSING

    pool = mp.Pool(procs)
    
    for ids,gender in zip([ids_m,ids_f],['m','f']):
        with open('results','w') as out:
            for uid,result in tq(pool.imap_unordered(songs_per_artist,ids,chunksize=100),total=len(ids_f)+len(ids_m)):
                out.write("{}\t{}\t{}\n".format(uid,gender,result))
    
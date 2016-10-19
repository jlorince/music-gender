import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import time
from os.path import basename
import pickle
import datetime

### FILTERS
filter_gender = ['m','f']
filter_playcount = 1000

### SUPPORT FUNCTIONS

def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

### SUPPORT DATA  -- > Need to figure out how to only load this when we need it...
#features = np.load('P:/Projects/BigMusic/jared.data/artist-features-w2v-400-15.npy')
#mapping = pd.read_pickle('P:/Projects/BigMusic/jared.data/artist-map-w2v-200-15.pkl').set_index('id')
distance_matrix = np.load('P:/Projects/BigMusic/jared.git/music-gender/data/w2v-400-15-distance_matrix-100k.npy')
idx_dict = pickle.load(open('P:/Projects/BigMusic/jared.git/music-gender/data/idx_dict_100k'))


### ANALYSIS FUNCTIONS

"""
Normalized number of unique artists (n_unique_artists / n_listens)
"""
def unique_artists_norm(fi):
    df =  parse_df(fi)
    return len(df['artist_id'].unique()) / float(len(df))

"""
Normalized number of unique songs (n_unique_songs / n_listens)
"""
def unique_songs_norm(fi):
    df =  parse_df(fi)
    return len(df['song_id'].unique()) / float(len(df))

"""
Time on site (timestamp of final listen - timestamp of first listen)
"""
def total_time(fi):
    df =  parse_df(fi,include_time = True)
    return (df.iloc[-1]['ts']-df.iloc[0]['ts']).total_seconds() / 86400.

"""
Artist listening distribution (propotion of listening allocated to most-listened, second-most-listend, etc. artists )
"""
def artist_rank_dist(fi):
    df = parse_df(fi)
    result = df['artist_id'].value_counts() / float(len(df))
    return result.values

"""
New artist encounter: For each scrobble, is this the user's first time listening to that ARTIST?
"""
def new_artist(fi):
    df = parse_df(fi)
    result = []
    encountered = set()
    for a in df['artist_id']:
        if a not in encountered:
            result.append(1)
            encountered.add(a)
        else:
            result.append(0)
    return np.array(result,dtype=float)

"""
New song encounter: For each scrobble, is this the user's first time listening to that SONG?
"""
def new_song(fi):
    df = parse_df(fi)
    result = []
    encountered = set()
    for s in df['song_id']:
        if s not in encountered:
            result.append(1)
            encountered.add(s)
        else:
            result.append(0)
    return np.array(result,dtype=float)

"""
Gini coefficient (over songs)
"""
def gini_songs(fi):
    df = parse_df(fi)
    return gini(df['song_id'].value_counts().values.astype(float))

"""
Gini coefficient (over artists)
"""
def gini_artists(fi):
    df = parse_df(fi)
    return gini(df['artist_id'].value_counts().values.astype(float))

"""
Calculate distance-based diversity on a user's set of unique artists
"""

def artist_diversity(fi):
    start = time.time()
    df = parse_df(fi)
    unique_artists = df['artist_id'].unique()
    indices = mapping.ix[unique_artists].dropna()['idx'].values.astype(int)
    feature_arr = features[indices]
    distances = pdist(feature_arr,metric='cosine')
    print "User {} done ({})".format(basename(fi),str(datetime.timedelta(seconds=(time.time()-start))))
    return np.mean(distances)

"""
Calculate distance-based diversity on all of a user's listens
"""
def diversity(fi):
    start = time.time()
    df = parse_df(fi)
    df['idx'] = df['artist_id'].apply(lambda x: idx_dict.get(x))
    df = df.dropna()
    artist_counts = df['idx'].value_counts().sort_index()
    n = len(df)
    result = ((artist_counts.values[:,None]*artist_counts.values) * distance_matrix[artist_counts.index.values][:,artist_counts.index.values]).sum() / float((n*n-1))
    print "User {} done ({})".format(basename(fi),str(datetime.timedelta(seconds=(time.time()-start))))
    return result




if __name__ == '__main__':

    import sys
    import multiprocessing as mp
    from glob import glob
    import math
    import time,datetime
    import os
    
    ### WRAPPER
    func_dict_single_value = {'unique_artists_norm':unique_artists_norm,'unique_songs_norm':unique_songs_norm,'total_time':total_time,'gini_songs':gini_songs,'gini_artists':gini_artists,'artist_diversity':artist_diversity,'diversity':diversity}
    func_dict_series_mean = {'artist_rank_dist':artist_rank_dist,'new_song':new_song,'new_artist':new_artist}
    combined = func_dict_single_value.copy()
    combined.update(func_dict_series_mean)
    
    func_name = sys.argv[1]
    if len(sys.argv)>2:
        extra_args = sys.argv[2:]
    func = combined.get(func_name)
    
    if func is None:
        raise("Must specify a valid function")

    if func == 'diversity':
        n_procs = 24
    else:
        n_procs = mp.cpu_count()
    pool = mp.Pool(n_procs)


    ### METADATA HANDLING
    user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'])

    user_data['sample_playcount'][user_data['sample_playcount']=='\\N'] = 0 
    user_data['sample_playcount'] = user_data['sample_playcount'].astype(int)

    filtered = user_data.loc[(user_data['gender'].isin(filter_gender)) & (user_data['sample_playcount']>=filter_playcount)][['user_id','gender']]

    ids_f = set(filtered[filtered['gender']=='f']['user_id'].astype(str))
    ids_m = set(filtered[filtered['gender']=='m']['user_id'].astype(str))


    files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')

    files_m = sorted([f for f in files if f[f.rfind('\\')+1:f.rfind('.')] in ids_m],key=os.path.getsize,reverse=True)
    files_f = sorted([f for f in files if f[f.rfind('\\')+1:f.rfind('.')] in ids_f],key=os.path.getsize,reverse=True)

    ### RUN MAIN PROCESSING
    if func_name in func_dict_single_value:
        for gender in ('m','f'):
            start = time.time()
            files = vars()['files_{}'.format(gender)]
            chunksize = int(math.ceil(len(files) / float(n_procs)))
            result = np.array(pool.map(func,files,chunksize=chunksize),dtype=str)
            with open('results/{}_{}'.format(func_name,gender),'w') as fout:
                fout.write('\n'.join(result))
            print "{} stats done ({})".format(gender,str(datetime.timedelta(seconds=(time.time()-start))))
    
    elif func_name in func_dict_series_mean:
        for gender in ('m','f'):
            start = time.time()
            files = vars()['files_{}'.format(gender)]
            total_files = len(files)
            chunksize = int(math.ceil(total_files / float(n_procs)))
            total = 0
            max_length = 0

            n = np.zeros(0,dtype=float)
            mean = np.zeros(0,dtype=float)
            M2 = np.zeros(0,dtype=float)

            for result in pool.imap_unordered(func,files,chunksize=chunksize):
                l = len(result)
                if l>max_length:
                    n = np.pad(n,(0,l-max_length),mode='constant',constant_values=0.)
                    mean = np.pad(mean,(0,l-max_length),mode='constant',constant_values=0.)
                    M2 = np.pad(M2,(0,l-max_length),mode='constant',constant_values=0.)
                    current = result
                    max_length = l

                else:
                    current = np.pad(result,(0,max_length-l),mode='constant',constant_values=[np.nan])

                mask = np.where(~np.isnan(current))
                n[mask]+=1
                delta = (current-mean)[mask]
                mean[mask] += delta/n[mask]
                M2[mask] += delta*(current-mean)[mask]
                total += 1
                print "{}/{} files processed ({})".format(total,total_files,gender)

            print "{} stats done ({})".format(gender,str(datetime.timedelta(seconds=(time.time()-start))))

            std = np.sqrt(M2 / (n - 1))
            np.savez('results/{}_{}.npz'.format(func_name,gender),mean=mean,std=std,n=n)

    pool.close()


    # result_f = pool.map(f,files_f)
    # with open('results/{}_f'.format(func),'w') as fout:
    #     fout.write('\n'.join(result_f))
    # result_m = pool.map(f,files_m)
    # with open('results/{}_m'.format(func),'w') as fout:
    #     fout.write('\n'.join(result_m))
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import time
from os.path import basename
import pickle
import datetime
from glob import glob
"""
set HADOOP_HOME=U:\Users\jjl2228\hadoop
set PYSPARK_DRIVER_PYTHON="ipython"
"""

### FILTERS
filter_gender = ['m','f']
filter_playcount = 1000

### SUPPORT FUNCTIONS

def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])


distance_matrix = np.load('P:/Projects/BigMusic/jared.git/music-gender/data/w2v-400-15-distance_matrix-100k.npy')
idx_dict = pickle.load(open('P:/Projects/BigMusic/jared.git/music-gender/data/idx_dict_100k'))


### ANALYSIS FUNCTIONS

"""
Calculate distance-based diversity on all of a user's listens
"""
def diversity(fi):
    start = time.time()
    df = parse_df(fi)
    df['idx'] = df['artist_id'].apply(lambda x: idx_dict.get(x))
    df = df.dropna()
    df['idx'] = df['idx'].astype(int)
    artist_counts = df['idx'].value_counts().sort_index()
    n = len(df)
    count_arr = artist_counts.values
    idx_arr = artist_counts.index.values
    #result = ((count_arr[:,None]*count_arr.values) * distance_matrix[idx_arr][:,idx_arr]).sum() / float(n*(n-1))
    result = ((count_arr[:,None]*count_arr.values) * np.array(dist_dict[i][idx_arr] for i in idx_arr)).sum() / float(n*(n-1))
    print "User {} done ({})".format(basename(fi),str(datetime.timedelta(seconds=(time.time()-start))))
    return result


### METADATA HANDLING
user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'])

user_data['sample_playcount'][user_data['sample_playcount']=='\\N'] = 0 
user_data['sample_playcount'] = user_data['sample_playcount'].astype(int)

filtered = user_data.loc[(user_data['gender'].isin(filter_gender)) & (user_data['sample_playcount']>=filter_playcount)][['user_id','gender']]

ids_f = set(filtered[filtered['gender']=='f']['user_id'].astype(str))
ids_m = set(filtered[filtered['gender']=='m']['user_id'].astype(str))


files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')

files_m = sc.parallelize(sorted([f for f in files if f[f.rfind('\\')+1:f.rfind('.')] in ids_m],key=os.path.getsize,reverse=True)[:10])
files_f = sc.parallelize(sorted([f for f in files if f[f.rfind('\\')+1:f.rfind('.')] in ids_f],key=os.path.getsize,reverse=True)[:10])

start = time.time()
result_m = files_m.map(diversity).collect()
print str(datetime.timedelta(seconds=(time.time()-start)))



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

    if func_name == 'diversity':
        n_procs = 12
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
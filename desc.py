import pandas as pd

### FILTERS
filter_gender = ['m','f']
filter_playcount = 1000

### SUPPORT FUNCTIONS

def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])

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
def artist_dist(fi):
    df = parse_df(fi)
    result = df['artist_id'].value_counts() / float(len(df))
    return result.values




if __name__ == '__main__':

    import sys
    import multiprocessing as mp
    from glob import glob
    import numpy as np
    
    n_procs = mp.cpu_count()
    pool = mp.Pool(n_procs)

    ### WRAPPER
    func_dict_single_value = {'unique_artists_norm':unique_artists_norm,'unique_songs_norm':unique_songs_norm,'total_time':total_time}
    func_dict_series_mean = {'artist_dist':artist_dist}
    combined = func_dict_single_value.copy()
    combined.update(func_dict_series_mean)
    
    func_name = sys.argv[1]
    if len(sys.argv)>2:
        extra_args = sys.argv[2:]
    func = combined.get(func_name)
    
    if func is None:
        raise("Must specify a valid function")


    ### METADATA HANDLING
    user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'])

    user_data['sample_playcount'][user_data['sample_playcount']=='\\N'] = 0 
    user_data['sample_playcount'] = user_data['sample_playcount'].astype(int)

    filtered = user_data.loc[(user_data['gender'].isin(filter_gender)) & (user_data['sample_playcount']>=filter_playcount)][['user_id','gender']]

    ids_f = set(filtered[filtered['gender']=='f']['user_id'].astype(str))
    ids_m = set(filtered[filtered['gender']=='m']['user_id'].astype(str))


    files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')

    files_m = [f for f in files if f[f.rfind('\\')+1:f.rfind('.')] in ids_m]
    files_f = [f for f in files if f[f.rfind('\\')+1:f.rfind('.')] in ids_f]

    ### RUN MAIN PROCESSING
    if func_name in func_dict_single_value:
        for gender in ('m','f'):
            files = vars()['files_{}'.format(gender)]
            chunksize = int(math.ceil(len(files) / float(n_procs)))
            result = np.array(pool.map(func,files,chunksize=chunksize),dtype=str)
            with open('results/{}_{}'.format(func_name,gender),'w') as fout:
                fout.write('\n'.join(result))
    
    elif func_name in func_series_mean:
        for gender in ('m','f'):
            files = vars()['files_{}'.format(gender)]
            total_files = len(files)
            chunksize = int(math.ceil(total_files / float(n_procs)))
            total = 0
            max_length = 0

            n = np.zeros(0,dtype=float)
            mean = np.zeros(0,dtype=float)
            M2 = np.zeros(0,dtype=float)

            for result in pool.imap_unordered(func,files,chunksize=chunksize):
                n = len(result)
                if n>max_length:
                    n = np.pad(n,(0,n-max_length),mode='constant',constant_values=0.)
                    mean = np.pad(mean,(0,n-max_length),mode='constant',constant_values=0.)
                    M2 = np.pad(M2,(0,n-max_length),mode='constant',constant_values=0.)
                    current = result
                    max_length = n

                else:
                    current = np.pad(result,(0,max_length-n),mode='constant',constant_values=[np.nan])

                mask = np.where(~np.isnan(current))
                n[mask]+=1
                delta = (data-mean)[mask]
                mean[mask] += delta/n[mask]
                M2[mask] += delta*(data-mean)[mask]
                total += 1
                print "{}/{} files processed ({})".format(total,total_files,gender)

                print "Stats done: {} ({})".format(gender,total)

            std = np.sqrt(M2 / (n - 1))
            np.savez('results/{}_{}.npz'.format(func_name,gender),mean=mean,std=std,n=n)

    pool.close()


    # result_f = pool.map(f,files_f)
    # with open('results/{}_f'.format(func),'w') as fout:
    #     fout.write('\n'.join(result_f))
    # result_m = pool.map(f,files_m)
    # with open('results/{}_m'.format(func),'w') as fout:
    #     fout.write('\n'.join(result_m))
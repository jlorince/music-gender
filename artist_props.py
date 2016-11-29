import pandas as pd
import multiprocessing as mp

filter_gender = ['m','f']
filter_playcount = 1000

artist_map = pd.read_pickle('P:/Projects/BigMusic/jared.data/artist-map-w2v-200-15.pkl').sort_values('idx')[:10000]
d = dict(zip(artist_map['id'],artist_map['idx']))

def process(fi):
    df = pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['artist_id'])
    df['idx'] = df['artist_id'].apply(lambda x: d.get(x,-1))
    vc = df.idx.value_counts()
    vc = vc / float(vc.sum())
    print fi[fi.rfind('\\')+1:fi.rfind('.')],

    return vc.reindex(xrange(10000),fill_value=0).values




if __name__=='__main__':
    from glob import glob
    import math
    import os

    user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'])

    user_data['sample_playcount'][user_data['sample_playcount']=='\\N'] = 0 
    user_data['sample_playcount'] = user_data['sample_playcount'].astype(int)

    filtered = user_data.loc[(user_data['gender'].isin(filter_gender)) & (user_data['sample_playcount']>=filter_playcount)][['user_id','gender','sample_playcount']]

    ids_f = set(filtered[filtered['gender']=='f']['user_id'])
    ids_m = set(filtered[filtered['gender']=='m']['user_id'])


    files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')

    files_m = sorted([f for f in files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_m],key=os.path.getsize,reverse=True)
    files_f = sorted([f for f in files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_f],key=os.path.getsize,reverse=True)

    procs = mp.cpu_count()
    pool = mp.Pool(procs)

    chunksize = int(math.ceil(len(files_m)/procs))
    result_m = np.vstack(pool.map(process,files_m,chunksize=chunksize))

    chunksize = int(math.ceil(len(files_f)/procs))
    result_f = np.vstack(pool.map(process,files_f,chunksize=chunksize))

    np.save('P:/Projects/BigMusic/jared.git/music-gender/data/artist_props_m.npy',result_m)
    np.save('P:/Projects/BigMusic/jared.git/music-gender/data/artist_props_f.npy',result_f)


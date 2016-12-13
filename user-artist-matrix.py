import pandas as pd
import multiprocessing as mp
import numpy as np
import os
import time,datetime
from scipy import sparse
from glob import glob

# by_gender=True
# d = 'P:/Projects/BigMusic/jared.git/music-gender/data/'

by_gender=False
d = 'P:/Projects/BigMusic/jared.data/'

class timed(object):
    def __init__(self,desc='command',pad='',**kwargs):
        self.desc = desc
        self.kwargs = kwargs
        self.pad = pad
    def __enter__(self):
        self.start = time.time()
        print '{} started...'.format(self.desc)
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            print '{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad)
        else:
            print '{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad)

def parse_df(fi,include_time=False):
    return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['artist_id'])

artist_map = pd.read_pickle('P:/Projects/BigMusic/jared.data/artist-map-w2v-200-15.pkl').sort_values('idx')[:10000]
#mapping = dict(zip(artist_map['id'],artist_map['idx']))
new_idx = artist_map['id'].values


def process(fi):
    df = parse_df(fi)
    vc = df['artist_id'].value_counts()
    vc = vc.reindex(new_idx,fill_value=0)
    return np.array(vc.values).reshape((10000,1))

if __name__=='__main__':
    import math

    with timed('file list setup'):

        all_files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')
        ids = [f[f.rfind('\\')+1:-4] for f in all_files]

        if by_gender:
            gender_data = pd.read_table('P:/Projects/BigMusic/jared.data/user_gender')
            ids_m = set(gender_data[gender_data['gender']=='m']['user_id'])
            ids_f = set(gender_data[gender_data['gender']=='f']['user_id'])

            
            files_m = sorted([f for f in all_files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_m],key=os.path.getsize,reverse=True)
            files_f = sorted([f for f in all_files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_f],key=os.path.getsize,reverse=True)
            n_m = len(files_m)
            n_f = len(files_f)

    with timed('pool setup'):
        start = time.time()
        n_procs = mp.cpu_count()
        pool = mp.Pool(n_procs)

    if by_gender:
    
        with timed('main run (males)'):
            chunksize = int(math.ceil(len(files_m) / float(n_procs)))
            result_m = np.hstack(pool.map(process,files_m,chunksize=chunksize))
            print result_m.shape

        with timed('main run (females)'):
            chunksize = int(math.ceil(len(files_f) / float(n_procs)))
            result_f = np.hstack(pool.map(process,files_f,chunksize=chunksize))
            print result_f.shape

        with timed('saving data'):
            d = 'P:/Projects/BigMusic/jared.git/music-gender/data/'
            np.save(d+'user-artist-matrix-m.npy',result_m)
            np.save(d+'user-artist-matrix-f.npy',result_f)

    else:

        with timed('main run'):
            chunksize = int(math.ceil(len(all_files) / float(n_procs)))
            result = np.hstack(pool.map(process,all_files,chunksize=chunksize))
            print result.shape
        with timed('saving data'):
            np.save(d+'user-artist-matrix-complete.npy',result)
            with open(d+'user-artist-matrix-id-idx','w') as out:
                out.write('\n'.join(ids)+'\n')




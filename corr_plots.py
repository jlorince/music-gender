import pandas as pd
import multiprocessing as mp
import numpy as np
import os
import time,datetime
from scipy import sparse
from glob import glob


# mapping = {}
# for line in open('P:/Projects/BigMusic/jared.gcc/artist_idx_feature_map'):
#     line = line.strip().split()
#     mapping[int(line[0])] = int(line[1])

artist_map = pd.read_pickle('P:/Projects/BigMusic/jared.data/artist-map-w2v-200-15.pkl').sort_values('idx')[:10000]
mapping = dict(zip(artist_map['id'],artist_map['idx']))


def parse_df(fi,include_time=False):
    return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['artist_id'])


def process(fi,thresh=10):
    df = parse_df(fi)
    result = []
    if thresh is not None:
        vc = df['artist_id'].value_counts()
        overthresh = vc[vc>=10]
        for a in overthresh.index:
            idx = mapping.get(a)
            if idx is not None:
                result.append(idx)
    else:
        for a in df['artist_id'].unique():
            idx = mapping.get(a)
            if idx is not None:
                result.append(idx)

    #print fi
    return np.array(result)



#files = sorted(glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*'),key=os.path.getsize,reverse=True)

# result = sparse.lil_matrix((len(files),len(mapping)))
# for i,fi in enumerate(files):
#     print i,fi
#     indices = process(fi)
#     result[i,indices] = 1

# mat = result.tocsr()

# co = mat.T.dot(mat)

if __name__=='__main__':
    import math

    print 'getting file lists...'
    start = time.time()

    gender_data = pd.read_table('P:/Projects/BigMusic/jared.data/user_gender')
    ids_m = gender_data[gender_data['gender']=='m']
    ids_f = gender_data[gender_data['gender']=='f']

    all_files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')
    files_m = sorted([f for f in all_files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_m],key=os.path.getsize,reverse=True)
    files_f = sorted([f for f in all_files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_f],key=os.path.getsize,reverse=True)
    n_m = len(files_m)
    n_f = len(files_f)
    print '...done in {}'.format(str(datetime.timedelta(seconds=(time.time()-start))))

    print 'Initializing pool...'
    start = time.time()
    n_procs = mp.cpu_count()
    pool = mp.Pool(n_procs)
    print '...done in {}'.format(str(datetime.timedelta(seconds=(time.time()-start))))

    for gender in ('m','f'):
        files = vars()['files_{}'.format(gender)]
        print 'Initializing matrix...'
        start = time.time()
        chunksize = int(math.ceil(len(files) / float(n_procs)))
        mat = np.zeros((len(files),len(mapping)))
        print '...done in {}'.format(str(datetime.timedelta(seconds=(time.time()-start))))

        print 'Processing files...'
        start = time.time()
        for i,arr in enumerate(pool.imap_unordered(process,files,chunksize=chunksize)):
            if len(arr)>0:
                mat[i,arr] = 1
            print "{}/{}".format(i+1,vars()['n_{}'.format(gender)])
        print '...done in {}'.format(str(datetime.timedelta(seconds=(time.time()-start))))

        print "Generating cooccurrence matrix ({})".format(gender)
        start = time.time()
        co = mat.T.dot(mat)
        print '...done in {}'.format(str(datetime.timedelta(seconds=(time.time()-start))))

        np.save('p:/Projects/BigMusic/jared.git/music-gender/comat-{}.npy'.format(gender),co)





"""
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
"""



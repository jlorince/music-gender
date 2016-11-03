import numpy as np
import time, datetime
from collections import Counter
import multiprocessing as mp    
import itertools
import sys
import pandas as pd

null_model_path = 'P:/Projects/BigMusic/jared.git/music-gender/data/NULL-MODELS/'

def parse(input_path):
    mean = np.zeros((10000,10000),dtype=float)
    M2 = np.zeros((10000,10000),dtype=float)
    n=0.
    for i in xrange(1000):
        n+=1
        current = np.load(null_model_path+'null-simple-{:04d}.npy'.format(model_idx))
        delta = current - mean
        mean += (delta / n)
        M2 += delta*(current-mean)
    np.save(null_model_path+'null-simple-mean.npy',mean)
    np.save(null_model_path+'null-simple-std.npy',np.sqrt(M2 / (n-1)))


def go(model_idx):
    thresh = 10

    user_scrobble_counts = [int(line.strip()) for line in open('P:/Projects/BigMusic/jared.data/user_scrobble_counts')]
    n_users = len(user_scrobble_counts)

    # get info for top 10k artists
    artist_data = pd.read_table('U:/Users/jjl2228/Desktop/artist_data',header=None,names=['artist_id','artist_name','scrobbles','listeners'])
    artist_map = pd.read_pickle('P:/Projects/BigMusic/jared.data/artist-map-w2v-200-15.pkl').sort_values('idx')[:10000]
    d = dict(zip(artist_map['id'],artist_map['idx']))
    print "Artist data {:04d} loaded".format(model_idx)

    # build raw data structure (sequence of artists)
    included = artist_data[artist_data.artist_id.isin(artist_map['id'])]
    data = np.ones(artist_data['scrobbles'].sum(),dtype=int)
    data = data * -1
    idx = 0
    for i,(aid,n) in enumerate(zip(included['artist_id'],included['scrobbles'])):
        artist_idx = d.get(aid)
        if artist_idx is not None:
            data[idx:idx+n] = artist_idx
        idx += n
    print "Base data structure {:04d} generated".format(model_idx)

    
    start = time.time()
    #np.random.seed(int(time.time()))
    
    # randomize
    shuf_start = time.time()
    np.random.shuffle(data)
    print "Data {:04d} shuffled in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-shuf_start))))

    # Generate base matrix
    mat_start = time.time()
    mat = np.zeros((n_users,10000))
    idx = 0
    for i,n_scrobbles in enumerate(user_scrobble_counts):       
        arr = data[idx:idx+n_scrobbles]
        c = Counter(arr[arr>-1])
        indices = [k for k in c if c[k]>=thresh]
        idx += n_scrobbles
        if len(indices)>0: mat[i][indices] = 1
    print "Matrix {:04d} generated in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-mat_start))))

    co_start = time.time()
    co = mat.T.dot(mat)
    print "Co-occurrence matrix {:04d} generated in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-co_start))))

    np.save('P:/Projects/BigMusic/jared.git/music-gender/data/NULL-MODELS/null-simple-{:04d}.npy'.format(model_idx),co)

    print "Null model {:04d} complete in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-start))))

    return None

if __name__=='__main__':


    print 'Starting parallel computations:'
    n_procs = 30
    pool = mp.Pool(n_procs)

    #pool.map(go,itertools.izip(xrange(1000),itertools.repeat(data)))
    pool.map(go,xrange(1000))





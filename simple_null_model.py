import numpy as np
import time, datetime
from collections import Counter
import multiprocessing as mp    
import itertools
import sys
import pandas as pd
import os
import signal

null_model_path = 'P:/Projects/BigMusic/jared.git/music-gender/data/NULL-MODELS/'

if __name__ != '__main__':
    thresh = 10

    #user_scrobble_counts = [int(line.strip()) for line in open('P:/Projects/BigMusic/jared.data/user_scrobble_counts')]
    user_scrobble_counts = pd.read_table('P:/Projects/BigMusic/jared.data/user_scrobble_counts_by_gender')
    gc = user_scrobble_counts['gender'].value_counts()
    m_count = gc.ix['m']
    f_count = gc.ix['f']

    # get info for top 10k artists
    #artist_data = pd.read_table('U:/Users/jjl2228/Desktop/artist_data',header=None,names=['artist_id','artist_name','scrobbles','listeners'])
    artist_scrobble_counts = pd.read_table('P:/Projects/BigMusic/jared.data/artist_scrobble_counts_by_gender')
    artist_map = pd.read_pickle('P:/Projects/BigMusic/jared.data/artist-map-w2v-200-15.pkl').sort_values('idx')[:10000]
    d = dict(zip(artist_map['id'],artist_map['idx']))
    print "Artist data loaded"

    # build raw data structure (sequence of artists)
    #included = artist_data[artist_data.artist_id.isin(artist_map['id'])]
    included = artist_scrobble_counts[artist_scrobble_counts.artist_id.isin(artist_map['id'])]
    #data = np.ones(artist_data['scrobbles'].sum(),dtype=int)
    data = np.ones(artist_scrobble_counts['total'].sum(),dtype=int)
    data = data * -1
    idx = 0
    for i,(aid,n) in enumerate(zip(included['artist_id'],included['total'])):
        artist_idx = d.get(aid)
        if artist_idx is not None:
            data[idx:idx+n] = artist_idx
        idx += n
    print "Base data structure generated"

def parse():
    mean = np.zeros((10000,10000),dtype=float)
    M2 = np.zeros((10000,10000),dtype=float)
    n=0.
    for i in xrange(1000):
        try:
            current = np.load(null_model_path+'null-simple-{:04d}.npy'.format(i))
            print i,
        except: 
            continue
        n+=1
        delta = current - mean
        mean += (delta / n)
        M2 += delta*(current-mean)
    np.save(null_model_path+'null-simple-mean.npy',mean)
    np.save(null_model_path+'null-simple-std.npy',np.sqrt(M2 / (n-1)))    

def go(model_idx):
    if os.path.exists('{}null-simple-m-{:04d}.npy'.format(null_model_path,model_idx)) and \
       os.path.exists('{}null-simple-f-{:04d}.npy'.format(null_model_path,model_idx)):
         print '{} already done - skipping'.format(model_idx)
         return None
  
    start = time.time()
    #np.random.seed(int(time.time()))
    
    # randomize
    shuf_start = time.time()
    np.random.shuffle(data)
    print "Data {:04d} shuffled in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-shuf_start))))

    # Generate base matrix
    idx = 0
    for gender in ('m','f'):
        if os.path.exists('{}null-simple-{}-{:04d}.npy'.format(null_model_path,gender,model_idx)):
            print '{} - {} already done - skipping'.format(model_idx,gender)
            continue
        mat_start = time.time()
        mat = np.zeros((globals()[gender+'_count'],10000))
        
        for i,n_scrobbles in enumerate(user_scrobble_counts[user_scrobble_counts['gender']==gender]['sample_playcount']):       
            arr = data[idx:idx+n_scrobbles]
            c = Counter(arr[arr>-1])
            indices = [k for k in c if c[k]>=thresh]
            idx += n_scrobbles
            if len(indices)>0: mat[i][indices] = 1
        print "Matrix {:04d} ({}) generated in {}".format(model_idx,gender,str(datetime.timedelta(seconds=(time.time()-mat_start))))

        co_start = time.time()
        co = mat.T.dot(mat)
        print "Co-occurrence matrix {:04d} ({}) generated in {}".format(model_idx,gender,str(datetime.timedelta(seconds=(time.time()-co_start))))

        np.save('{}null-simple-{}-{:04d}.npy'.format(null_model_path,gender,model_idx),co)

    print "Null model {:04d} complete in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-start))))

    return None

def main(n_procs):
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_procs)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        #pool.map(go,xrange(1000))
        res = pool.map_async(go,xrange(1000))
        res.get(9999999999999999)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        print("Normal termination")
        pool.close()
    pool.join()

if __name__ == '__main__':

    n_procs = int(sys.argv[1])
    main(n_procs)

    # print 'Starting parallel computations:'
    # pool = mp.Pool(n_procs)

    # #pool.map(go,itertools.izip(xrange(1000),itertools.repeat(data)))
    # try:
    #     pool.map(go,xrange(1000))
    # except KeyboardInterrupt:
    #     pool.terminate()
    #     pool.join()





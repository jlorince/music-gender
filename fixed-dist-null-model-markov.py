import numpy as np
import time, datetime
from collections import Counter
import multiprocessing as mp    
import itertools
import sys
import pandas as pd
import os
import signal
from scipy.sparse import csr_matrix

#null_model_path = 'P:/Projects/BigMusic/jared.git/music-gender/data/NULL-MODELS/'
null_model_path = 'S:/UsersData/jjl2228/NULL-MODELS/'


def parse(combine_genders=False,mode='comat'):

    if mode  == 'markov':
        prefix = 'markov-'
    else:
        prefix =''

    if combine_genders:
        iter_over = ('combined',)
    else:
        iter_over = ('m','f')
    for gender in iter_over:
        if mode == 'comat':
            mean = np.zeros((10000,10000),dtype=float)
            M2 = np.zeros((10000,10000),dtype=float)

        elif mode == 'markov':
            mean = np.zeros((10001,10001),dtype=float)
            M2 = np.zeros((10001,10001),dtype=float)
        
        n=0.
        for i in xrange(1000):
            try:
                if combine_genders:
                    m = np.load(null_model_path+'{}null-artist_dist-m-{:04d}.npy'.format(prefix,i))
                    f = np.load(null_model_path+'{}null-artist_dist-f-{:04d}.npy'.format(prefix,i))
                    current = m+f
                else:
                    current = np.load(null_model_path+'{}null-artist_dist-{}-{:04d}.npy'.format(prefix,gender,i))
                print i,

                if mode == 'markov':
                    current = current / current.sum(1,keepdims=True)
            except: 
                continue
            n+=1
            delta = current - mean
            mean += (delta / n)
            M2 += delta*(current-mean)

        np.save(null_model_path+'{}null-artist_dist-{}-mean.npy'.format(prefix,gender),mean)
        np.save(null_model_path+'{}null-artist_dist-{}-std.npy'.format(prefix,gender),np.sqrt(M2 / (n-1)))




def markov(model_idx,mode):
    exists_m = os.path.exists('{}markov-null-artist_dist-m-{:04d}.npy'.format(null_model_path,model_idx))
    exists_f = os.path.exists('{}markov-null-artist_dist-f-{:04d}.npy'.format(null_model_path,model_idx))

    if exists_m and exists_f:
        print '{} already done - skipping'.format(model_idx)
        return None

    genders = ('m','f')
  
    start = time.time()
    #np.random.seed(int(time.time()))
    
    # randomize
    shuf_start = time.time()
    if mode == 'artist':
        user_arr = data['user'].values.copy()
        np.random.shuffle(user_arr)
        data['user'] = user_arr
    elif mode == 'user':
        artist_arr = data['artist'].values.copy()
        np.random.shuffle(artist_arr)
        data['artist'] = artist_arr
    data = data.sort_values(by=['gender','user'],ascending=True)
    print "Data {:04d} shuffled in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-shuf_start))))

    # Generate base matrix    
        
    last_user = data['user'].iloc[0]
    indices = []
    last_gender = 'f'

    mat_start = time.time()
    mat = np.zeros((10001,10001))
    
    for i,row in data.iterrows():

        if row.gender != last_gender:

            np.save('{}markov-null-artist_dist-{}-{:04d}.npy'.format(null_model_path,gender,model_idx),co)

            last_user = row.user
            indices = []
            last_gender = 'm'

            mat_start = time.time()
            mat = np.zeros((10001,10001))

        if row.user != last_user:
            np.random.shuffle(indices)
            df = pd.DataFrame({'idx':indices})
            df['idx'][df['idx']==-1] = 10000
            df['last'] = df['idx'].shift(1)
            df.set_value(0,'last',10000)
            df['transition'] = df.apply(lambda row: (int(row['last']),row['idx']) ,axis=1)
            vc = df['transition'].value_counts()

            for (i,j),n in vc.iteritems():
                mat[i,j] += n

            indices = []        
        
        indices += [row.artist]*row.N

        last_gender = row.gender
        last_user = row.user
   

    print "Null model {:04d} complete in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-start))))

    return None

def main(n_procs,mode):
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_procs)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        res = pool.map_async(markov,itertools.izip(xrange(1000),itertools.repeat(mode)))
        res.get(9999999999999999)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        print("Normal termination")
        pool.close()
    pool.join()

if __name__ != '__main__':

    data = pd.read_table('P:/Projects/BigMusic/jared.data/user_artist_scrobble_counts_by_gender_idx10k',
                          header=None,names=['user','gender','artist','N'])

    user_scrobble_counts = pd.read_table('P:/Projects/BigMusic/jared.data/user_scrobble_counts_by_gender')
    gc = user_scrobble_counts['gender'].value_counts()
    m_count = gc.ix['m']
    f_count = gc.ix['f']





if __name__ == '__main__':

    n_procs = int(sys.argv[1])
    mode = sys.argv[2]
    main(n_procs,mode)

    # print 'Starting parallel computations:'
    # pool = mp.Pool(n_procs)

    # #pool.map(go,itertools.izip(xrange(1000),itertools.repeat(data)))
    # try:
    #     pool.map(go,xrange(1000))
    # except KeyboardInterrupt:
    #     pool.terminate()
    #     pool.join()





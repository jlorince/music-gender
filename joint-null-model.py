import multiprocessing as mp
import time,sys,os
import pandas as pd
import numpy as np

thresh = 10

if __name__ != "__main__":

    start = time.time()
    chunk_size = 10000

    artist_map = pd.read_pickle('P:/Projects/BigMusic/jared.data/artist-map-w2v-200-15.pkl').sort_values('idx')[:10000]
    d = dict(zip(artist_map['id'],artist_map['idx']))
    print "Artist data loaded"

    df = pd.read_table('P:/Projects/BigMusic/jared.data/user_artist_scrobble_counts_by_gender',header=None,names=['user','gender','artist','n'])
    total_edges = len(df)

    counts_by_edge_weight = df['n'].value_counts()


    chunks = []
    for weight,cnt in counts_by_edge_weight.iteritems():
        if cnt>=chunk_size:
            current = df[df['n']==weight]
            current['artist'] = current['artist'].apply(lambda x: d.get(x,-1))
            chunks.append(current)
        else:
            break

    idx = 0
    condensed = df.join(counts_by_edge_weight[counts_by_edge_weight<chunks_size],on='n',rsuffix='_').sort_values('n')
    while idx<len(condensed):
        current = condensed.iloc[idx:idx+chunk_size]
        current['artist'] = current['artist'].apply(lambda x: d.get(x,-1))
        chunks.append()
        idx += chunksize

    print 'Base data prepped in {}',format(str(datetime.timedelta(seconds=(time.time()-shuf_start))))


def comat(model_idx):
    exists_m = os.path.exists('{}null-joint-m-{:04d}.npy'.format(null_model_path,model_idx))
    exists_f = os.path.exists('{}null-joint-f-{:04d}.npy'.format(null_model_path,model_idx))

    if exists_m and exists_f:
        print '{} already done - skipping'.format(model_idx)
        return None

    genders = ('m','f')
  
    start = time.time()
    #np.random.seed(int(time.time()))
    
    # randomize
    shuf_start = time.time()
    for chunk in chunks:
        artist_arr = chunk['artist'].values.copy()
        np.random.shuffle(artist_arr)
        chunk['artist'] = artist_arr
    data = pd.concat(chunks)

    print "Data {:04d} shuffled in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-shuf_start))))

    for gender in genders:
        mat_start = time.time()
        mat = np.zeros((globals()[gender+'_count'],10000))

        result = data[(data['gender']==gender)&(data['N']>=thresh)].groupby('user').apply(lambda x: [a for a in x.artist.values if a != -1])

        # for i,indices in enumerate(result.values):
        #     mat[i,indices] = 1
        for i,user in enumerate({'m':males,'f':females}[gender]):
            indices = result.get(user)
            if indices is not None:
                mat[i,indices] = 1

        print "Matrix {:04d} ({}) generated in {}".format(model_idx,gender,str(datetime.timedelta(seconds=(time.time()-mat_start))))

        co_start = time.time()
        co = mat.T.dot(mat)
        print "Co-occurrence matrix {:04d} ({}) generated in {}".format(model_idx,gender,str(datetime.timedelta(seconds=(time.time()-co_start))))

        np.save('{}null-joint-{}-{:04d}.npy'.format(null_model_path,gender,model_idx),co)

    print "Null model {:04d} complete in {}".format(model_idx,str(datetime.timedelta(seconds=(time.time()-start))))

    return None

def main(n_procs,func):
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_procs)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        res = pool.map_async(comat,xrange(1000))

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




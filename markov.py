import pandas as pd
import multiprocessing as mp
import numpy as np
import os
import time,datetime
from scipy import sparse
from glob import glob
import signal
import sys
import math

if __name__ != '__main__':

    def save_sparse_csr(filename,array):
        np.savez(filename,data = array.data ,indices=array.indices,
                 indptr =array.indptr, shape=array.shape )

    def load_sparse_csr(filename):
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                             shape = loader['shape'])


    artist_map = pd.read_pickle('P:/Projects/BigMusic/jared.data/artist-map-w2v-200-15.pkl').sort_values('idx')[:10000]
    mapping = dict(zip(artist_map['id'],artist_map['idx']))


    def parse_df(fi,include_time=False):
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['artist_id'])


def process(fi):
    df = parse_df(fi)
    df['idx'] = df['artist_id'].apply(lambda x: mapping.get(x,10000))
    df['last'] = df['idx'].shift(1)
    df.set_value(0,'last',10000)
    df['transition'] = df.apply(lambda row: (int(row['last']),row['idx']) ,axis=1)
    vc = df['transition'].value_counts()
    i,j = zip(*vc.index.values)

    mat = sparse.csr_matrix((vc.values,(i,j)),shape=(10001,10001))

    save_sparse_csr('S:/UsersData/jjl2228/scratch/{}.npz'.format(fi[fi.find('\\')+1:-4]),mat)
    print "{} processed".format(fi)


def main(n_procs):
    all_files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_procs)
    chunksize = int(math.ceil(len(all_files) / float(n_procs)))
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        res = pool.map_async(process,all_files)
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


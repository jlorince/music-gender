import pandas as pd
#import multiprocessing as mp
import numpy as np
import os

from scipy import sparse
from glob import glob


mapping = {}
for line in open('P:/Projects/BigMusic/jared.gcc/artist_idx_feature_map'):
    line = line.strip().split()
    mapping[int(line[0])] = int(line[1])


def parse_df(fi,include_time=False):
    return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['artist_id'])


def process(fi):
    df = parse_df(fi)
    vc = df['artist_id'].value_counts()
    overthresh = vc[vc>=100]
    result = []
    for a in overthresh.index:
        idx = mapping.get(a)
        if idx is not None:
            result.append(idx)
    return np.array(result)


# if __name__=='__main__':
#     pool = mp.Pool(4)

files = sorted(glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*'),key=os.path.getsize,reverse=True)

result = sparse.lil_matrix((len(files),len(mapping)))
for i,fi in enumerate(files):
    print i,fi
    indices = process(fi)
    result[i,indices] = 1

mat = result.tocsr()

co = mat.T.dot(mat)



def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])




import numpy as np
import time, datetime
from collections import Counter
import multiprocessing as mp    
import itertools
import sys
import pandas as pd
import os
import signal

thresh = 10


#null_model_path = 'P:/Projects/BigMusic/jared.git/music-gender/data/NULL-MODELS/'
null_model_path = 'S:/UsersData/jjl2228/NULL-MODELS/'


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


def parse():

    for gender in ('m','f'):
        mean = np.zeros((10000,10000),dtype=float)
        M2 = np.zeros((10000,10000),dtype=float)
    
        n=0.
        for i in xrange(1000):
            try:
                if combine_genders:
                    m = np.load(null_model_path+'{}null-gender-shuffle-m-{:04d}.npy'.format(prefix,i))
                    f = np.load(null_model_path+'{}null-gender-shuffle-f-{:04d}.npy'.format(prefix,i))
                    current = m+f
                else:
                    current = np.load(null_model_path+'{}null-gender-shuffle-{}-{:04d}.npy'.format(prefix,gender,i))

                #current += 1 # pseudocount
                print i,

            except: 
                continue
            n+=1
            delta = current - mean
            mean += (delta / n)
            M2 += delta*(current-mean)

        np.save(null_model_path+'{}null-gender-shuffle-{}-mean.npy'.format(prefix,gender),mean)
        np.save(null_model_path+'{}null-gender-shuffle-{}-std.npy'.format(prefix,gender),np.sqrt(M2 / (n-1)))

def comat(model_idx):
    # exists_m = os.path.exists('{}null-gender-shuffle-m-{:04d}.npy'.format(null_model_path,model_idx))
    # exists_f = os.path.exists('{}null-simple-f-{:04d}.npy'.format(null_model_path,model_idx))

    # if exists_m and exists_f:
    #     print '{} already done - skipping'.format(model_idx)
    #     return None

    with timed("Null model {:04d}".format(model_idx)):

        # randomize
        np.random.shuffle(mask)
        # Generate base matrix

        data_m = combined[np.where(mask==1)[0]]
        data_f = combined[np.where(mask==0)[0]]
    
        # for gender,mat in zip(('m','f'),(data_m,data_f)):
        #     with timed("Co-occurrence matrices {:04d} ({})".format(model_idx,gender)):    
        #         co = mat.T.dot(mat)
        #         np.save('{}null-gender-shuffle-{}-{:04d}.npy'.format(null_model_path,gender,model_idx),co)
        with timed("Co-occurrence matrix {:04d} (m)".format(model_idx)):
            co_m = data_m.T.dot(data_m)
        with timed("Co-occurrence matrix {:04d} (f)".format(model_idx)):
            co_f = data_f.T.dot(data_f)

    return co_m,co_f

def main(n_procs):
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_procs)
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        #pool.map(comat,xrange(1000))
        #res.get(9999999999999999)

        with timed('mean/m2 setup'):
            mean_m = np.zeros((10000,10000),dtype=float)
            M2_m = np.zeros((10000,10000),dtype=float)
            mean_f = np.zeros((10000,10000),dtype=float)
            M2_f = np.zeros((10000,10000),dtype=float)

            n = 0

        for i,(co_m,co_f) in enumerate(pool.imap(comat,xrange(1000))):
            with timed('processing results for idx {}'.format(i)):
                n+=1
                delta_m = co_m - mean_m
                mean_m += (delta_m / n)
                M2_m += delta_m*(co_m-mean_m)

                delta_f = co_f - mean_f
                mean_f += (delta_f / n)
                M2_f += delta_f*(co_f-mean_f)

        with timed('saving male data'):
            np.save(null_model_path+'null-gender-shuffle-m-mean.npy',mean_m)
            np.save(null_model_path+'null-gender-shuffle-m-std.npy',np.sqrt(M2_m / (n-1)))

        with timed('saving female data'):
            np.save(null_model_path+'null-gender-shuffle-f-mean.npy',mean_f)
            np.save(null_model_path+'null-gender-shuffle-f-std.npy',np.sqrt(M2_f / (n-1)))


    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        print("Normal termination")
        pool.close()
    pool.join()

if __name__ != '__main__':

    with timed('initial data setup (pid={})'.format(os.getpid())):
        d = 'P:/Projects/BigMusic/jared.git/music-gender/data/'
        #d = '/backup/home/jared/music-gender/data/'
        counts_m = np.load(d+'user-artist-matrix-m.npy')
        counts_f = np.load(d+'user-artist-matrix-f.npy')

        n_m = counts_m.shape[1]
        n_f = counts_f.shape[1]

        combined = np.hstack([counts_m,counts_f])
        combined = (combined>=thresh).astype(float).T
        
        mask = np.concatenate([np.ones(n_m),np.zeros(n_f)])


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





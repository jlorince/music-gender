import numpy as np
import multiprocessing as mp
import itertools
import time,datetime

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


d = 'P:/Projects/BigMusic/jared.git/music-gender/data/'
counts_m = np.load(d+'user-artist-matrix-m.npy')
counts_f = np.load(d+'user-artist-matrix-f.npy')

dists_m = (counts_m / counts_m.sum(1,dtype=float,keepdims=True))
dists_f = (counts_f / counts_f.sum(1,dtype=float,keepdims=True))

def a_entropy(arr,alpha=2):
    return (1./(1.-alpha)) * ((arr**alpha).sum()-1.0)

def div(p,q,alpha=2):
    return a_entropy((p+q)/2.,alpha)-.5*(a_entropy(p,alpha))-.5*(a_entropy(q,alpha))

def div_max(p,q,alpha=2):
    return (((2.0**(1.-alpha)) - 1.0)/2.) * (a_entropy(p,alpha)+a_entropy(q,alpha)+(2./(1.-alpha)))

def div_norm(p,q,alpha=2):
    return div(p,q,alpha) / div_max(p,q,alpha)

def wrapper_f(tup):
    i_p,i_q = tup
    p = dists_f[i_p]
    q = dists_f[i_q]
    return div_norm(p,q,alpha=2)

def wrapper_m(tup):
    i_p,i_q = tup
    p = dists_m[i_p]
    q = dists_m[i_q]
    return div_norm(p,q,alpha=2)


if __name__=='__main__':
    import math
    from scipy.misc import comb

    procs = mp.cpu_count()
    pool = mp.Pool(procs)

    total_comps = comb(10000,2)
    chunksize = int(math.ceil(total_comps/procs))
    
    for gender in ('m','f'):

        f = {'m':wrapper_m,'f':wrapper_f}[gender]
        result = []

        for i,divergence in enumerate(pool.imap(f,itertools.combinations(10000,2),chunksize=chunksize),1):
            result.append(divergence)
            if i%100000==0:
                print "{}/{} ({:.2f}% complete)".format(i,int(total_comps),100*(i/total_comps))

        np.save('{}divergences_{}.npy'.format(gender),np.array(result))





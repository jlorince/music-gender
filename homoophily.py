import numpy as np
#import multiprocessing as mp
import pathos.multiprocessing as mp
import itertools
import time,datetime
import sys
import pandas as pd

total_comps = 100000

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


#d = 'P:/Projects/BigMusic/jared.git/music-gender/data/'
d = '/backup/home/jared/music-gender/'
counts_m = np.load(d+'user-artist-matrix-m.npy')
counts_f = np.load(d+'user-artist-matrix-f.npy')

combined = np.hstack([counts_m,counts_f])

dists = (combined / combined.sum(0,dtype=float,keepdims=True))

n = combined.shape[1]

friendship_links = set()
for line in open(d+'friendship-links-internal.txt'):
    a,b = map(int,line.strip().split())
    friendship_links.add((a,b))


def a_entropy(arr,alpha=2):
    return (1./(1.-alpha)) * ((arr**alpha).sum()-1.0)

def div(p,q,alpha=2):
    return a_entropy((p+q)/2.,alpha)-.5*(a_entropy(p,alpha))-.5*(a_entropy(q,alpha))

def div_max(p,q,alpha=2):
    return (((2.0**(1.-alpha)) - 1.0)/2.) * (a_entropy(p,alpha)+a_entropy(q,alpha)+(2./(1.-alpha)))

def div_norm(p,q,alpha=2):
    return div(p,q,alpha) / div_max(p,q,alpha)

def wrapper(tup):
    print tup
    a,b = tup
    p = dists[:,a]
    q = dists[:,b]
    divergence = div_norm(p,q,alpha=2)
    link = (a,b) in friendship_links
    return divergence,link



if __name__=='__main__':
    import math
    from scipy.misc import comb

    procs = mp.cpu_count()
    pool = mp.Pool(procs)

    with timed('comparison set generation'):
        s = set()
        cnt = 0
        while cnt<total_comps:
            a = np.random.randint(0,n)
            b = np.random.randint(0,n)
            if a>b:
                a,b = b,a
            elif b==a:
                continue
            comb = (a,b)

            if comb not in s:
                s.add(comb)
                cnt+=1

    chunksize = int(math.ceil(total_comps/procs))

    final = []
    for i,result in enumerate(pool.imap(wrapper,s,chunksize=chunksize),1):
        final.append(result)
        #if i%10000==0:
        print "{}/{} ({:.2f}% complete)".format(i,total_comps,100*(i/float(total_comps)))

    df = pd.DataFrame(final,columns=['divergence','link'])

    result = df.groupby(np.digitize(df['divergence'],bins=np.arange(0,1,.01))).link.mean()
    
    






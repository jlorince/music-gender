import numpy as np
#import multiprocessing as mp
#import pathos.multiprocessing as mp
import itertools
import time,datetime
import sys
import pandas as pd
from scipy.stats import entropy

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
#d = '/backup/home/jared/music-gender/data/'
d = 'P:/Projects/BigMusic/jared.data/'

#combined = np.load('/backup/home/jared/user-artist-matrix-complete.npy')

with timed('setting up user-artist matrix'):
    combined = np.load(d+'/user-artist-matrix-complete.npy')

    dists = (combined / combined.sum(0,dtype=float,keepdims=True))
    combined[np.isnan(combined)] = 0

    n = combined.shape[1]

with timed('loading link data'):
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

def jsd(p,q,b=2):
    m = (p+q)/2.
    return 0.5*entropy(p,m,base=b) + 0.5*entropy(q,m,base=b)

def wrapper(tup):
    #print tup
    a,b = tup
    p = dists[:,a]
    q = dists[:,b]
    #divergence = div_norm(p,q,alpha=2)
    divergence = jsd(p,q)
    link = (a,b) in friendship_links
    return divergence,int(link)


if __name__ == '__main__':

    #total_comps = int(sys.argv[1])
    total_comps = 10000000

    # import math

    # with timed('pool spinup'):
    #     procs = mp.cpu_count()
    #     pool = mp.Pool(procs)

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

    #chunksize = int(math.ceil(total_comps / float(procs)))

    # final = []
    # for i,result in enumerate(pool.imap(wrapper,s,chunksize=chunksize),1):
    #     final.append(result)
    #     #if i%10000==0:
    #     print "{}/{} ({:.2f}% complete)".format(i,total_comps,100*(i/float(total_comps)))
    with timed('main processing'):
        #final = pool.map(wrapper,s,chunksize=chunksize)
        final = []
        for i,tup in enumerate(s,1):
            final.append(wrapper(tup))
            if i%100000==0:
                print "{}/{} ({:.2f}% complete)".format(i,int(total_comps),100*(i/float(total_comps)))


    with timed('building dataframe'):
        df = pd.DataFrame(final,columns=['divergence','link'])

    with timed('grouping'):
        result = df.dropna().groupby(np.digitize(df['divergence'],bins=np.arange(0,1,.01))).link.describe().unstack()
        #result.to_pickle('P:/Projects/BigMusic/jared.data/homophily-data-sampled.pkl')
        #result.to_pickle(d+'homophily-data-sampled.pkl')








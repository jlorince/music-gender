import multiprocessing as mp
import sys,math,itertools,os
import pandas as pd
import numpy as np
from functools import partial
import scipy.stats 
from scipy.stats import entropy as scipy_ent


datadir = 'P:/Projects/BigMusic/jared.data/'

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


###############################
############ Shared ###########
###############################
with timed('sampling setup'):

    ## Generate per-bucket counts
    # user_artist_df = pd.read_table(datadir+'user_artist_scrobble_counts_by_gender_idx10k',header=None,names=['user_id','gender','artist','n'])
    # user_data = user_artist_df[user_artist_df.artist!=-1].groupby('user_id').agg({'gender':lambda x: x.iloc[0],'n':np.sum}).reset_index()
    # user_data['floor_logn'] = user_data.n.apply(lambda x: int(np.log10(x)))
    # bin_counts = user_data.floor_logn.value_counts()
    # bin_weights = bin_counts / float(bin_counts.sum())
    # maxbin = bin_counts.index.max()
    # sample_size = (len(user_data[(user_data.floor_logn==maxbin)&(user_data.gender=='f')]) / 10.) *2
    # perc_mult = sample_size / bin_weights.ix[maxbin]
    # bucket_sizes = bin_weights * perc_mult
    # per_bucket_gender_counts = np.round(bucket_sizes/2).astype(int).sort_index().values

    # ## Generate male and female playcounts
    # m_playcounts = user_data[user_data.gender=='m'].n
    # f_playcounts = user_data[user_data.gender=='f'].n
    
    #for d in ('f_playcounts','m_playcounts','per_bucket_gender_counts'):
    #    np.save("{}bowie_support/{}.npy".format(datadir,d),vars()[d])
    #user_data.to_pickle(datadir+'bowie_support/user_data.pkl')

    per_bucket_gender_counts = np.load(datadir+'bowie_support/per_bucket_gender_counts_10k.npy')
    m_playcounts = np.load(datadir+'bowie_support/m_playcounts_10k.npy')
    f_playcounts = np.load(datadir+'bowie_support/f_playcounts_10k.npy')
    user_data = pd.read_pickle(datadir+'bowie_support/user_data_10k.pkl')

## Generate empirical multinomial distribution
with timed('artist distribution setup'):
    #artist_counts = user_artist_df[user_artist_df.artist!=-1].groupby('artist').n.sum()
    #artist_probs = (artist_counts / float(artist_counts.sum()))
    #np.save("{}bowie_support/{}.npy".format(datadir,'artist_probs'),artist_probs)
    artist_probs = np.load(datadir+'bowie_support/artist_probs_10k.npy')


###############################
########## Bootstrap ##########
###############################

def create_pop_sample(seed=None):
    if seed is not None:
        np.random.seed(seed)
    #users = []
    user_playcounts = []
    for bucket,grp in user_data.groupby('floor_logn'):
        #users+=(list(np.random.choice(grp[grp.gender=='m'].user_id,per_bucket_gender_counts[bucket])))
        #users+=(list(np.random.choice(grp[grp.gender=='f'].user_id,per_bucket_gender_counts[bucket])))
        user_playcounts+=(list(np.random.choice(grp[grp.gender=='m'].n,per_bucket_gender_counts[bucket])))
        user_playcounts+=(list(np.random.choice(grp[grp.gender=='f'].n,per_bucket_gender_counts[bucket])))
    #return users
    return user_playcounts

def run_bootstrap(idx,mode):
    with timed('Running bootstrap idx {} ({})'.format(idx,mode)):
        #result = [[]]*len(funcs)
        result = [[] for _ in xrange(len(funcs))]
        #with timed('Getting playcounts (idx={})'.format(idx)):
        playcounts = {'m':m_playcounts,'f':f_playcounts,'n':create_pop_sample()}[mode]
        for u in playcounts:
            listening = np.random.multinomial(u,artist_probs).astype(float)
            listening = listening[listening>0]
            for i,f in enumerate(funcs):
                result[i].append(f(listening))
        return [np.mean(r) for r in result]

def calc_zscore(user_data,bs_data):
    user_id,df = user_data
    result = []
    for i,f in enumerate(funcs):
        value = f(df.n.values.astype(float))
        result += [value, (value-bs_data[i][0])/bs_data[i][1]]
    return [user_id,df.gender.iloc[0]]+result

###############################
########### Metrics ###########
###############################
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    #array = array.flatten() #all values are treated equally, arrays must be 1d
    #array = array[array>0]
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    #array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def unique_artists_norm(arr):
    return len(arr) / float(arr.sum())

def unique_artists(arr):
    return float(len(arr))

def entropy(arr):
    return scipy_ent(arr,base=2)

funcs = [unique_artists,unique_artists_norm,entropy,gini]

if __name__=='__main__':

    # try:
    #     funckeys = sys.argv[1:]
    #     funcs = [{'gini':gini,'unique_artists':unique_artists,'unique_artists_norm':unique_artists_norm,'entropy':entropy}[f] for f in funckeys]
    # except KeyError:
    #     raise Exception("Must provide a valid function name")
    with timed('loading user_artist_df'):
        user_artist_df = pd.read_table(datadir+'user_artist_scrobble_counts_by_gender_idx10k',header=None,names=['user_id','gender','artist','n'])
        user_artist_df = user_artist_df[user_artist_df.artist!=-1]
        #user_artist_df = pd.read_table(datadir+'user_artist_scrobble_counts_by_gender',header=None,names=['user_id','gender','artist','n'])

    
    procs = mp.cpu_count()
    pool = mp.Pool(procs)
    
    n_runs = 10000
    chunksize = int(math.ceil(n_runs / float(procs)))


    for mode in ('n','m','f'):
        # func_partial = partial(run_bootstrap,mode=mode)
        # with timed('running bootstrap, mode={}'.format(mode),pad='------'):
        #     results = zip(*pool.map(func_partial,xrange(n_runs),chunksize=chunksize))
        #     with open(datadir+'sampled_gender_results/raw_bootstrap_{}'.format(mode),'w') as out:
        #         bs_data = [(np.mean(r),np.std(r)) for r in results]            
        #         for i,r in enumerate(results):
        #             out.write(str(funcs[i]).split()[1] + '\t'+ ','.join(map(str,r)) + '\t' + str(bs_data[i][0]) + '\t' + str(bs_data[i][1]) + '\n')        

        current = pd.read_table('P:/Projects/BigMusic/jared.data/sampled_gender_results/raw_bootstrap_{}'.format(mode),header=None,names=['metric','values','mean','std'])
        bs_data = [(row['mean'],row['std']) for i, row in current.iterrows()]            

        with timed('generating z-scores, mode={}'.format(mode),pad='------'):
            func_partial = partial(calc_zscore,bs_data=bs_data)
            if mode =='n':
                zscores = pool.map(func_partial,user_artist_df.groupby('user_id'))
            else:
                zscores = pool.map(func_partial,user_artist_df[user_artist_df.gender==mode].groupby('user_id'),chunksize=chunksize)
            with open(datadir+'sampled_gender_results/{}_{}'.format('-'.join([str(f).split()[1] for f in funcs]),mode),'w') as fout:
                for result in zscores:
                    fout.write('\t'.join(map(str,result))+'\n')

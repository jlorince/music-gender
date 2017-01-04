import multiprocessing as mp
import sys,math,itertools,os
import pandas as pd
import numpy as np

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
        user_playcounts+=(list(np.random.choice(grp[grp.gender=='m'].sample_playcount,per_bucket_gender_counts[bucket])))
        user_playcounts+=(list(np.random.choice(grp[grp.gender=='f'].sample_playcount,per_bucket_gender_counts[bucket])))
    #return users
    return user_playcounts

def run_bootstrap(idx,f,mode):
    with timed('Running bootstrap idx {} ({}, {})'.format(idx,str(f).split()[1],mode)):
        result = []
        playcounts = {'m':m_playcounts,'f':f_playcounts,'n':create_pop_sample()}[mode]
        for u in playcounts:
            listening = np.random.multinomial(u,artist_probs)
            result.append(f(listening))
        return numpy.mean(result)

def calc_zscore(f,user_data):
    user_id,df = user_data
    return user_id,df.gender.iloc[0],(f(df.n)-bs_mean)/bs_std

###############################
########### Metrics ###########
###############################
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
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

if __name__=='__main__':

    try:
        funckey = sys.argv[1]
        func = {'gini':gini,'unique_artists':unique_artists,'unique_artists_norm':unique_artists_norm}[funckey]
    except KeyError:
        raise Exception("Must provide a valid function name")

    
    with timed('sampling setup'):

        ## Generate per-bucket counts
        user_data = pd.read_table('P:/Projects/BigMusic/jared.data/user_scrobble_counts_by_gender')
        user_data['floor_logn'] = user_data.sample_playcount.apply(lambda x: int(np.log10(x)))
        bin_counts = user_data.floor_logn.value_counts()
        bin_weights = bin_counts / float(bin_counts.sum())
        maxbin = bin_counts.index.max()
        sample_size = (len(user_data[(user_data.floor_logn==maxbin)&(user_data.gender=='f')]) / 10.) *2
        perc_mult = sample_size / bin_weights.ix[maxbin]
        bucket_sizes = bin_weights * perc_mult
        per_bucket_gender_count = np.round(bucket_sizes/2).astype(int).sort_index().values
        user_data = user_data.set_index('user_id')

        ## Generate male and female playcounts
        m_playcounts = user_data[user_data.gender=='m'].sample_playcount
        f_playcounts = user_data[user_data.gender=='f'].sample_playcount

    ## Generate empirical multinomial distribution
    with timed('artist distribution setup'):
        user_artist_df = pd.read_table('P:/Projects/BigMusic/jared.data/user_artist_scrobble_counts_by_gender',header=None,names=['user_id','gender','artist','n'])
        artist_counts = user_artist_df.groupby('artist').n.sum()
        artist_probs = (artist_counts / float(artist_counts.sum())).values

    procs = mp.cpu_count()
    pool = mp.Pool(procs)
    
    n_runs = 10000
    chunksize = int(math.ceil(n_runs / float(args.procs)))

    for mode in ('n','m','f'):
        func_partial = partial(run_bootstrap,f=func,mode='n')
        with timed('running bootstrap, mode={}'.format(mode),pad='------'):
            result = pool.map(func_partial,xrange(n_runs))
            bs_mean = np.mean(result)
            bs_std = np.std(result)

        with timed('generating z-scores, mode={}'.format(mode),pad='------'):
            if mode =='n':
                zscores = pool.map(calc_zscore,user_artist_df.groupby('user_id'))
            else:
                zscores = pool.map(calc_zscore,user_artist_df[user_artist_df.gender==mode].groupby('user_id'))
            with open('P:/Projects/BigMusic/gender_results/{}_{}'.format(funckey,mode),'w') as fout:
                for user,gender,zscore in zscore:
                    fout.write("{}\t{}\t{}\n".format(user,gender,zscore))









    # generate 'Bowie null'

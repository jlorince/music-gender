import sys,math,itertools,os
import pandas as pd
import numpy as np
from functools import partial
import scipy.stats 
from scipy.stats import entropy as scipy_ent
from mpi4py import MPI

###############################
### Config variables
###############################

funcs = [unique_artists,unique_artists_norm,entropy,gini]
datadir = '/projects/p30035/'
n_runs = 10

###############################
### Setup
###############################

# MPI variable setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
node = MPI.Get_processor_name()

# logfile setup
mode = MPI.MODE_WRONLY|MPI.MODE_CREATE#|MPI.MODE_APPEND
now = str(datetime.datetime.now())
logfile = MPI.File.Open(comm, "logfile_{}.log".format(now[:now.rfind('.')].replace(' ','|').replace(':','')), mode)
logfile.Set_atomicity(True)
def logger(msg):
    msg = "[node={}, rank={}] [{}] {}\n".format(node,rank,str(datetime.datetime.now()),msg)
    logfile.Write_shared(msg)


###############################
### Support functions
###############################
class timed(object):
    def __init__(self,desc='command',pad='',**kwargs):
        self.desc = desc
        self.kwargs = kwargs
        self.pad = pad
    def __enter__(self):
        self.start = time.time()
        if len(self.kwargs)==0:
            logger('{} started...'.format(self.desc))
        else:
            logger('{} ({}) started...'.format(self.desc,','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()])))
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            logger('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            logger('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad))


###############################
### Load shared data 
###############################
if rank == 0:
    with timed('loading shared data from disk'):

        per_bucket_gender_counts = np.load(datadir+'bowie_support/per_bucket_gender_counts.npy')
        m_playcounts = np.load(datadir+'bowie_support/m_playcounts.npy')
        f_playcounts = np.load(datadir+'bowie_support/f_playcounts.npy')
        user_data = pd.read_pickle(datadir+'bowie_support/user_data.pkl')
        artist_probs = np.load(datadir+'bowie_support/artist_probs.npy')
else:
    per_bucket_gender_counts = None
    m_playcounts = None
    f_playcounts = None
    user_data = None
    artist_probs = None

per_bucket_gender_counts = comm.bcast(per_bucket_gender_counts,root=0)
m_playcounts = comm.bcast(m_playcounts,root=0)
f_playcounts = comm.bcast(f_playcounts,root=0)
user_data = comm.bcast(user_data,root=0)
artist_probs = comm.bcast(artist_probs,root=0)


###############################
### Bootstrap functions
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
        for u in playcounts[:100]:
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
    #if np.amin(array) < 0:
    #    array -= np.amin(array) #values cannot be negative
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
    #return thoth.calc_entropy(arr,1000)[0]

###############################
### Null model calculation
###############################

runs_per_rank = int(math.ceil(n_runs / float(size)))
start_idx = runs_per_rank * rank


log = open('log','w')
for mode in ('n','m','f'):

    result = [run_bootstrap(i,mode=mode) for i in xrange(start_idx,min(n_runs,start_idx+runs_per_rank))]


    gathered = comm.gather(result,root=0):
    if rank ==0:
        combined = []
        for r in gathered:
            combined += r
        results = zip(*combined)
        bs_data = [(np.mean(r),np.std(r)) for r in results]

        for i,r in enumerate(results):
            log.write(str(funcs[i]).split()[1] + '\t'+ ','.join(map(str,r)) + '\t' + str(bs_data[i][0]) + '\t' + str(bs_data[i][1]) + '\n') 

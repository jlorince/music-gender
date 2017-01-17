import multiprocessing as mp
import sys,math,itertools,os
import pandas as pd
import numpy as np
from functools import partial
import scipy.stats 
from scipy.stats import entropy as scipy_ent
from tqdm import tqdm,trange


datadir = 'P:/Projects/BigMusic/jared.data/'
measures = ['total_listens','unique_artists','unique_artists_norm', 'entropy', 'gini']

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



def sampler(gender):
    users = []
    for bucket,grp in user_data[user_data.gender==gender].groupby('floor_logn'):
        users += list(np.random.choice(grp.user_id,per_bucket_gender_counts[bucket]))
    data = result_df.ix[users]

    # for measure in ['total_listens','unique_artists','unique_artists_norm', 'entropy', 'gini']:
    #     vars()['hist_'+measure] = np.histogram(gender[measure],bins=bins[measure])[0]
    # break

    return [np.histogram(data[measure],bins=bins[measure])[0]  for measure in ['total_listens','unique_artists','unique_artists_norm', 'entropy', 'gini']]



###############################
########### Metrics ###########
###############################
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    #array = array.flatten() #all values are treated equally, arrays must be 1d
    #array = array[array>0]
    # if np.amin(array) < 0:
    #     array -= np.amin(array) #values cannot be negative
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

def total_listens(arr):
    return arr.sum()

def wrapper(df):
    arr = df.n.values.astype(float)
    return pd.Series([df.iloc[0].user_id]+[globals()[measure](arr) for measure in measures], index = ['user_id']+measures)
    #return pd.Series([df.iloc[0].user_id,arr.sum(),unique_artists(arr),unique_artists_norm(arr),entropy(arr),gini(arr)],index=['user_id','total_listens','unique_artists','unique_artists_norm', 'entropy', 'gini'])




with timed('loading  data'):
    user_artist_df = pd.read_table(datadir+'user_artist_scrobble_counts_by_gender',header=None,names=['user_id','gender','artist','n'])
    user_data = pd.read_pickle(datadir+'bowie_support/user_data.pkl')
    #user_ids, grouped_dfs = zip(*[g for g in user_artist_df.groupby('user_id',sort=False)])

    ## Generate per-bucket counts
    # user_data = user_artist_df[user_artist_df.artist!=-1].groupby('user_id').agg({'gender':lambda x: x.iloc[0],'n':np.sum}).reset_index()
    # user_data['floor_logn'] = user_data.n.apply(lambda x: int(np.log10(x)))
    bin_counts = user_data.floor_logn.value_counts()
    bin_weights = bin_counts / float(bin_counts.sum())
    maxbin = bin_counts.index.max()
    sample_size = (len(user_data[(user_data.floor_logn==maxbin)&(user_data.gender=='f')]) / 5.) *2
    perc_mult = sample_size / bin_weights.ix[maxbin]
    bucket_sizes = bin_weights * perc_mult
    per_bucket_gender_counts = np.round(bucket_sizes/2).astype(int).sort_index().values

# procs = mp.cpu_count()
# pool = mp.Pool(procs)

# n_runs = 10000
# chunksize_samples = int(math.ceil(n_runs / float(procs)))
# chunksize_users = int(math.ceil(len(user_ids) / float(procs)))
    
#results = pool.map(wrapper,grouped_dfs,chunksize=chunksize_users)
#result_df = pd.DataFrame(results,columns = ['unique_artists','unique_artists_norm', 'entropy', 'gini'])
tqdm.pandas()
result_df = user_artist_df.groupby('user_id',sort=False).progress_apply(wrapper).set_index('user_id')
#desc = result_df.describe()
bins = {}
for col in measures:
    #bins[col] = np.linspace(desc[col]['min'],desc[col]['max'],1000)
    bins[col] = np.histogram(result_df[col],bins='auto')[1]

final_results_m = []
final_results_f = []
for _ in trange(1000):
    final_results_m.append(sampler('m'))
    final_results_f.append(sampler('f'))

final_m = {}
final_f = {}
final_results_m = zip(*final_results_m)
final_results_f = zip(*final_results_f)
for i,measure in enumerate(tqdm(measures)):
    final_m[measure] = np.percentile(np.vstack(final_results_m[i]),[5,50,95],axis=0)
    final_f[measure] = np.percentile(np.vstack(final_results_f[i]),[5,50,95],axis=0)

from matplotlib import pyplot as plt
import seaborn
colors = seaborn.color_palette()

def plotter(m,logy=False,xlim=None):
    fig,ax = plt.subplots()
    n = len(bins[m])-1


    for i,gender in enumerate((final_m,final_f)):
        ax.plot(gender[m][1],color=colors[i])
        ax.fill_between(range(n),gender[m][0],gender[m][2],alpha=.25,color=colors[i])
        if logy:
            ax.set_yscale('log')
        if xlim:
            ax.set_xlim(xlim)

    plt.show()


joined = user_data.join(result_df,on='user_id')
fig,axes = plt.subplots(1,3,figsize=(12,4))
label_dict = {'gini':'Gini Coefficient (higher = lower diversity)','entropy':'Entropy (bits, higher = greater diversity)','unique_artists':'Unique Artists Listened'}
for  i,(m,ax) in enumerate(zip(['gini','entropy','unique_artists'],axes)):
    bins = np.logspace(0,np.log10(joined.n.max()),100)
    #bins = np.linspace(0,100000,101,True)
    joined['bin'] = np.digitize(joined.n,bins)#,right=True)
    counts = joined.groupby(['bin','gender'])[m].count().unstack()
    counts.index = bins[counts.index-1]
    props = counts.apply(lambda x: x/x.sum())
    means = joined.groupby(['bin','gender'])[m].mean().unstack()
    means.index = bins[means.index-1]
    if i==0:
        means.plot(ax=ax,style=['-','--'])
    else:
        means.plot(ax=ax,legend=False,style=['-','--'])
    #props.plot(ax=ax)
    se = joined.groupby(['bin','gender'])[m].apply(lambda x: 1.96*(x.std()/np.sqrt(len(x)))).unstack()
    se.index = bins[se.index-1]
    ax.fill_between(se.index,(means-se)['f'],(means+se)['f'],alpha=0.25,color=colors[0])
    ax.fill_between(se.index,(means-se)['m'],(means+se)['m'],alpha=0.25,color=colors[1])
    #ax.set_xscale('log')
    

    xlims = np.percentile(joined.n,5),np.percentile(joined.n,95)
    trimmed = means.ix[bins[int(np.digitize(xlims[0],bins))]:bins[int(np.digitize(xlims[1],bins))]]
    ylims = trimmed.min().min(),trimmed.max().max()*1.1
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    #ax.set_yscale('log')
    ax.set_title(label_dict[m])
    ax.set_xlabel('Total listens')
plt.tight_layout()
fig.savefig('U:/Users/jjl2228/Desktop/netsci.pdf',bbox_inches='tight')
plt.show()
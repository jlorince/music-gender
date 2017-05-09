from annoy import AnnoyIndex
import random,time,itertools,os,pickle,glob,argparse,datetime,warnings
import numpy as np
import pandas as pd
import multiprocess as mp
from tqdm import tqdm as tq
from collections import Counter

help_string="""

THIS CODE IS ONLY TESTED AGAINST PYTHON 3.6!!!

lde.py --> Local Density Estimation script

Generates estiamtes of local density in a feature space by computing the mean distance between songs and their nearest neighbors, leveraging the annoy approximate nearest neighbors library.

For each song, computes the mean distance to its top k nearest neighbors in each year bin, as well as the temporal distribution of those top k neighbors.

Results are saved in a directory specifed by the following:
result_path = args.result_dir+'_'.join([str(v) for v in [{args.d2v_params OR 'echonest'},args.index_seed,args.knn,args.trees,args.search_k,args.min_songs,args.songs_per_bin]])+'/'

Use the index_seed argument to load an existing global-norm specified by the provided values of `trees` and `index_seed`. Otherwise this will always generate a new index.

Model assumes there is a file `index_years.npy` in the datadir that specifies the release year for each song.

Annoy documentation: https://github.com/spotify/annoy
Doc2Vec documentation: https://radimrehurek.com/gensim/models/doc2vec.html
"""

class timed(object):
    def __init__(self,desc='command',pad='',**kwargs):
        self.desc = desc
        self.kwargs = kwargs
        self.pad = pad
    def __enter__(self):
        self.start = time.time()
        print('{} started...'.format(self.desc))
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            print('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            print('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad))

# annoy returns euclidean distance of normed vector
# (i.e. sqrt(2-2*cos(u, v))), so this just converts
# to standard cosine distance
def convert_distance(d):
   return (d**2) /2
vec_convert_distance = np.vectorize(convert_distance)


# computes LDE w.r.t to each year, as well as the neighbor distribution across years
# ONLY VALID when index type is `global` or `global-norm`
# query can be an integer (look up trained item) or  vector (look up held out item)
def lde (query):
    if type(query) in (np.int64,np.int32,int):
        neighbors,distances = t.get_nns_by_item(query, args.knn+1, search_k=args.search_k, include_distances=True)
    elif type(query) in (np.core.memmap,np.ndarray):
        neighbors,distances = t.get_nns_by_vector(query, args.knn, search_k=args.search_k, include_distances=True)
    else:
        raise Exception("Invalid input")
    neighbors = np.array(neighbors[1:])
    neighbor_bins = bin_indices_sampled[neighbors]
    d_result = np.repeat(np.nan,nbins)
    n_result = np.zeros(nbins,dtype=int)
    for b in allowed_bins:
        idx = np.where(neighbor_bins==b)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            d_result[bin_dict[b]] = convert_distance(np.mean([distances[i] for i in idx]))
        n_result[bin_dict[b]] = len(idx)
    return d_result,n_result


def lde_wrapper(b):
    filename = "bin_{}_{}-{}".format(b,args.bins[b-1],args.bins[b]-1)
    with open('{}results_{}'.format(result_path,filename),'w') as out:
        current = np.where(bin_indices_sampled==b)[0]
        current_untrained = untrained[np.where(bin_indices[untrained]==b)[0]]
        total = len(current)+len(current_untrained)
        done = 0
        for song in current:
            if done%100==0:
                print("{}: {}/{} ({:.2f}%)".format(b,done,total,100*(done/total)))
                out.flush()
            d,n = lde(song)
            out.write("{}\t{}\t{}\n".format(indices[song],','.join(map(str,d)),','.join(map(str,n))))
            done +=1
        for song in current_untrained:
            if done%100==0:
                print("{}: {}/{} ({:.2f}%)".format(b,done,total,100*(done/total)))
                out.flush()
            d,n = lde(features[song])
            out.write("{}\t{}\t{}\n".format(song,','.join(map(str,d)),','.join(map(str,n))))
            done +=1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(help_string)
    parser.add_argument("--d2v_params",help="*IF* using d2v model features, specify d2v model paramter in format 'size-window-min_count-sample', e.g. '100-5-5-0' (see gensim doc2vec documentation for details of these parameters), so we know which subfolder to look in. If not provided, script will assume it's working with echonest sonic features, and will attempt to load `echonest_features.npy` from datadir",default=None,type=str)
    parser.add_argument("--bins", help="Specfiy the year binning in format: start_year-end_year-bin_width, such data will be binned for years >= start_year and < end_year (and limited to songs released in that range). E.g. the argument '1980-2000-5' will define bins 1980-1984, 1985-1989, ..., 1990-1994, 1995-1999",default='1970-2015-5')
    parser.add_argument("--index_dir", help="Where annoy index files are located. Defaults to same directory as d2v model files",default=None)
    parser.add_argument("--index_seed", help="Specify loading a model with this seed. Only useful if doing multiple runs and we want to run against a particular randomly seeded model. If unspecified a new model will be generated.",default=None)
    parser.add_argument("--datadir",help="path to data directory",default='/backup/home/jared/echonest_features/',type=str)
    parser.add_argument("--procs",help="Specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("--knn",help="number of nearest neighbors to be used in density computations, default=1000",default=1000,type=int)
    parser.add_argument("--trees",help="number of projection trees for knn index, default=100 (see annoy documentation)",default=100,type=int)
    parser.add_argument("--search_k",help="search_k paramter for knn index, default = `trees`*`knn` (see annoy documentation)",default=None,type=int)
    parser.add_argument("--min_songs",help="Minimum number of songs that must be included within a bin (defaults to 50k",default=20000,type=int)
    parser.add_argument("--result_dir",help="Output directory for results. A subfolder in this directory (named with relevant params) will be created here. By default folder is generated inside datadir.",default=None,type=str)
    args = parser.parse_args()

    ### ARGUMENT SETUP

    if args.search_k is None:
        args.search_k = args.trees * args.knn
    if args.index_dir is None:
        if args.d2v_params is not None:
            args.index_dir = args.datadir+args.d2v_params+'/'
        else:
            args.index_dir = args.datadir

    if args.result_dir is None:
        args.result_dir = args.datadir
    
    #bin setup
    bin_args = [i for i in map(int,args.bins.split('-'))]
    bin_args[1]+=1
    args.bins = np.arange(*bin_args)

    # index years contains the release year for each song
    index_years = np.load(args.datadir+'index_years.npy')

    # we mmap these to facilitate parallel computations
    if args.d2v_params is not None:
        features = np.load('{0}{1}/model_{1}.wv.syn0.npy'.format(args.datadir,args.d2v_params),mmap_mode='r')
    else:
        features = np.load('{0}echonest_features.npy'.format(args.datadir),mmap_mode='r')

    # get dimensionality for index
    f = features.shape[1]

    t = AnnoyIndex(f,metric='angular')

    ### this block handles all the binning/sampling

    bin_indices = np.digitize(index_years,args.bins)
    unique_bins,bin_counts = np.unique(bin_indices,return_counts=True) 
    def legal_bin(i,b):
        if (b==0) or (b>=len(args.bins)):
            return False
        if bin_counts[i] < args.min_songs:
            return False
        return True
    allowed_bins = [b for i,b in enumerate(unique_bins) if legal_bin(i,b)]
    nbins = len(allowed_bins)
    args.songs_per_bin = bin_counts[allowed_bins].min()
    bin_dict = {b:i for i,b in enumerate(allowed_bins)}

    if args.index_seed is not None:
        try:
            t.load('{}index_norm_{}_{}.ann'.format(args.index_dir,args.trees,args.index_seed))
            indices = np.load('{}index_norm_{}_{}.ann.indices.npy'.format(args.index_dir,args.trees,args.index_seed))
        except FileNotFoundError:
            raise Exception('You have specified an invalid seed (file does not exist)')

    else:

        indices = []
        idx = 0
        args.index_seed = np.random.randint(999999)
        print('----RANDOM SEED = {}----'.format(args.index_seed))
        np.random.seed(args.index_seed)

        for b in tq(allowed_bins):
            idx_current = np.random.choice(np.where(bin_indices==b)[0],args.songs_per_bin,replace=False)
            indices.append(idx_current)
            for vec in tq(features[idx_current]):
                t.add_item(idx, vec)
                idx+=1

        indices = np.concatenate(indices)
        np.save('{}index_norm_{}_{}.ann.indices'.format(args.index_dir,args.trees,args.index_seed),indices)

        with timed('building index'):
            t.build(args.trees) 
        with timed('saving index'):
            t.save('{}index_norm_{}_{}.ann'.format(args.index_dir,args.trees,args.index_seed))

    bin_indices_sampled = bin_indices[indices]
    untrained = np.ones(len(bin_indices),dtype=int)
    untrained[indices] = 0
    untrained[np.where(~np.in1d(bin_indices,allowed_bins))[0]]=0

    if args.d2v_params is not None:
        result_path = args.result_dir+'_'.join([str(v) for v in [args.d2v_params,args.index_seed,args.knn,args.trees,args.search_k,args.min_songs,args.songs_per_bin]])+'/'
    else:
        result_path = args.result_dir+'_'.join([str(v) for v in ['echonest',args.index_seed,args.knn,args.trees,args.search_k,args.min_songs,args.songs_per_bin]])+'/'
    if os.path.exists(result_path):
        raise Exception("Result directory already exists!!")
    os.mkdir(result_path)


    pool = mp.Pool(min(args.procs,len(allowed_bins)))
    pool.map(lde_wrapper,allowed_bins)
    pool.terminate()
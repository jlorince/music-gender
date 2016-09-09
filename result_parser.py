import numpy as np
#import argparse
import pandas as pd
import sys


gender_mapping = pd.read_table('/N/u/jlorince/BigRed2/gender-mapping.txt',header=None,names=['user','gender'])
rootdir = sys.argv[1]
if len(sys.argv)>2:
    gender_to_process = (sys.argb[2],)
else:
    gender_to_process = ('f','m')
#rootdir = '/N/dc2/scratch/jlorince/new_artist_discovery_rate_by_gender/'
#rootdir = '/N/dc2/scratch/jlorince/simple_artist_diversity_by_gender/'

max_length = 50000

# first pass to get max length
"""
include = set()
for gender in ('m','f'):
    mx = 0
    total = 0
    for user in gender_mapping[gender_mapping['gender']==gender]['user']:
        #
        try:
            n = open('/N/dc2/scratch/jlorince/new_artist_discovery_rate_by_gender/{}'.format(user)).read().strip().split(',')
            include.add(user)
            if n>mx:
                mx = n
        except IOError:
            continue
        total+=1
        print "Max length checks: {} ({})".format(gender,total)
    vars()['mx_{}'.format(gender)] = mx
"""

# now pad and generate our means and CIs

for gender in gender_to_process:
    total = 0

    #mx = vars()['mx_{}'.format(gender)]

    n = np.zeros(max_length,dtype=float)
    mean = np.zeros(max_length,dtype=float)
    M2 = np.zeros(max_length,dtype=float)

    for user in gender_mapping[gender_mapping['gender']==gender]['user']:
        #if user not in include:
        #    continue
        try:
            data = np.loadtxt(rootdir+'{}'.format(user),delimiter=',')[:max_length]
        except IOError:
            continue
        data = np.pad(data,(0,max_length-len(data)),mode='constant',constant_values=[np.nan])


        mask = np.where(~np.isnan(data))
        n[mask]+=1
        delta = (data-mean)[mask]
        mean[mask] += delta/n[mask]
        M2[mask] += delta*(data-mean)[mask]
        total += 1

        print "Stats done: {} ({})".format(gender,total)

    std = np.sqrt(M2 / (n - 1))
    np.savez('{}_results'.format(gender),mean=mean,std=std,n=n)
    #mean,np.sqrt(M2 / (n - 1))



#if __name__ == '__main__':

 #   parser = argparse.ArgumentParser("Need to add some more documentation")

    #parser.add_argument("-f", "--file",default=None,type=str)
    #parser.add_argument("-r", "--resultdir",default=None,type=str)

import numpy as np
import pandas as pd
import glob
from scipy import sparse

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f,1):
            pass
    return i

artist_data = pd.read_table('U:/Users/jjl2228/Desktop/artist_data',header=None,names=['artist_id','artist_name','scrobbles','listeners'])
artist_counts = dict(zip(artist_data['artist_id'],artist_data['scrobbles']))

included = artist_data[(artist_data['scrobbles']>=1000)&(artist_data['listeners']>=100)]
data = np.zeros(artist_data['scrobbles'].sum(),dtype=int)
idx = 0
for i,(aid,n) in enumerate(zip(included['artist_id'],included['scrobbles'])):
    data[idx:idx+n] = aid
    idx += n
    #print i
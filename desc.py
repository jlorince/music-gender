from pathos import multiprocessing as mp
from glob import glob

### FILTERS
filter_gender = ['m','f']
filter_playcount = 1000


### METADATA HANDLING
user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'])

user_data['sample_playcount'][user_data['sample_playcount']=='\\N'] = 0 
user_data['sample_playcount'] = user_data['sample_playcount'].astype(int)

filtered = user_data.loc[(user_data['gender'].isin(filter_gender)) & (user_data['sample_playcount']>=filter_playcount)][['user_id','gender']]

ids_f = set(filtered[filtered['gender']=='f']['user_id'].astype(str))
ids_m = set(filtered[filtered['gender']=='m']['user_id'].astype(str))


files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')

files_m = [f for f in files if f[f.rfind('\\')+1:f.rfind('.')] in ids_m]
files_f = [f for f in files if f[f.rfind('\\')+1:f.rfind('.')] in ids_f]

### SUPPORT FUNCTIONS

def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])

### ANALYSIS FUNCTIONS

"""
Normalized number of unique artists (n_unique_artists / n_listens)
"""
def unique_artists_norm(fi):
    df =  parse_df(fi)
    return len(df['artist_id'].unique())


pool 
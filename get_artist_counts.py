import pandas as pd

def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])

user_scrobble_counts = pd.read_table('P:/Projects/BigMusic/jared.data/user_scrobble_counts_by_gender')
n = len(user_scrobble_counts)

cnts_m = {}
cnts_f = {}


for i,(user,gender) in enumerate(zip(user_scrobble_counts['user_id'],user_scrobble_counts['gender']),1):
    print "{}/{}".format(i,n)
    vc = parse_df('P:/Projects/BigMusic/jared.iu/scrobbles-complete/{}.txt'.format(user))['artist_id'].value_counts()
    for artist,cnt in vc.iteritems():
        vars()['cnts_{}'.format(gender)][artist] = vars()['cnts_{}'.format(gender)].get(artist,0) + cnt

all_artists = set(cnts_m.keys()+cnts_f.keys())

with open('P:/Projects/BigMusic/jared.data/artist_scrobble_counts_by_gender','w') as fout:
    fout.write("artist_id\tm\tf\ttotal\n")
    for a in all_artists:  
        m = cnts_m.get(a,0)
        f = cnts_f.get(a,0)
        fout.write("{}\t{}\t{}\t{}\n".format(a,m,f,m+f))

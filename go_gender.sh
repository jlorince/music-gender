#Wrapper sript for running analysis on BigRed2

rootdir='/N/u/jlorince/BigRed2/music-gender/'


#### New artist discovery rate

python ${rootdir}exploratory_analyses.py -f $1 -r /N/dc2/scratch/jlorince/new_artist_discovery_rate_by_gender/

#### Simple artist diversity over listens
python ${rootdir}exploratory_analyses.py -f $1 -r /N/dc2/scratch/jlorince/simple_artist_diversity_by_gender/

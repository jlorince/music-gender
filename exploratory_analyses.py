import pandas as pd
import argparse
import numpy as np
import datetime
import logging


class gender_stuff(object):

    # init just takes in command line arguments and sets up logging
    def __init__(self,args,logging_level=logging.INFO):

        self.args = args

        # logger setup
        now = datetime.datetime.now()
        log_filename = now.strftime('setup_%Y%m%d_%H%M%S.log')
        logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
        self.rootLogger = logging.getLogger()

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)
        self.rootLogger.setLevel(logging_level)

        self.df = pd.read_table(self.args.file,header=None,names=['artist_id','ts'])

        self.filename = self.args.file.split('/')[-1]
        self.user = self.filename[:self.filename.find('.')]


    def rolling_new_artist_mean(self,window_size=100):

        result = []
        encountered = set()
        for a in self.df['artist_id']:
            if a not in encountered:
                result.append(1)
                encountered.add(a)
            else:
                result.append(0)
        self.df['new'] = result

        output = self.df['new'].rolling(window=window_size).mean()

        with open(self.args.resultdir+self.user,'w') as fout:
            fout.write(','.join(output[window_size-1:].fillna('').values.astype(str))+'\n')
        self.rootLogger.info('User {} processed successfully ({})'.format(self.user,self.filename))

    def rolling_artist_diversity(self,window_size=100):
        output = self.df['artist_id'].rolling(window=100).aggregate(lambda x: len(set(x))/float(len(x)))
        with open(self.args.resultdir+self.user,'w') as fout:
            fout.write(','.join(output[window_size-1:].fillna('').values.astype(str))+'\n')
        self.rootLogger.info('User {} processed successfully ({})'.format(self.user,self.filename))


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Need to add some more documentation")

    parser.add_argument("-f", "--file",default=None,type=str)
    parser.add_argument("-r", "--resultdir",default=None,type=str)

    args = parser.parse_args()

    g = gender_stuff(args)
    #g.rolling_new_artist_mean()
    g.rolling_artist_diversity()



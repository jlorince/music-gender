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


    @staticmethod
    def userFromFile(fi):
        #return fi.split('/')[-1].split('_')[-1][:-4]
        filename = fi.split('/')[-1]
        return filename[filename.find('.')]

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

        output = pd.rolling_mean(self.df['new'],window=window_size)

        with open(self.args.resultdir+user,'w') as fout:
            fout.write(','.join(output.fillna('').values.astype(str))+'\n')
        self.rootLogger.info('User {} processed successfully ({})'.format(user,fi))

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Need to add some more documentation")

    parser.add_argument("-f", "--file",default=None,type=str)
    parser.add_argument("-r", "--resultdir",default=None,type=str)

    args = parser.parse_args()

    g = gender_stuff(args)
    g.rolling_new_artist_mean()



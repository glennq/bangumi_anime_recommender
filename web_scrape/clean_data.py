import numpy as np
import pandas as pd
import requests
import os


def clean_data(fpath):
    df = pd.read_csv(os.path.join(fpath, 'ratings_all.csv'))
    df.SubId = np.object_(np.int64(df.SubId))
    df.UserId = np.object_(df.UserId)
    df = df[df.Rating != 'email__']
    df.Rating = np.int64(df.Rating)
    print 'finished basic cleaning'
    cnt = 0
    uniq_subid = df.SubId.unique()
    for i in uniq_subid:
        req = requests.get('http://bgm.tv/subject/' + str(i))
        if req.url.split('/')[-1] != str(i):
            df = df[df.SubId != i]
        cnt += 1
        print 'finished {}/{}'.format(cnt, len(uniq_subid))

    df.to_csv(os.path.join(fpath, 'ratings.csv'), index=False)


def main():
    fpath = os.path.join(os.pardir, os.pardir, 'data')
    clean_data(fpath)


if __name__ == '__main__':
    main()

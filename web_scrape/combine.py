import numpy as np
import pandas as pd
import os


def combine_data(fpath):
    args = zip(range(1, 125000, 2500), range(2500, 125001, 2500))
    cnames = np.array(['DateTime', 'SubId', 'UserId', 'Rating'])
    data = pd.DataFrame(columns=cnames)
    for i in args:
        fname = 'ratings_{}_{}.csv'.format(i[0], i[1])
        try:
            df = pd.read_csv(os.path.join(fpath, fname),
                             header=None, names=cnames)
        except IOError:
            print 'Could not read file {}, continue with others'.format(fname)
            continue
        data = data.append(df)
    data.SubId = np.int64(data.SubId)
    data.to_csv(os.path.join(fpath, 'ratings_all.csv'), index=False)


def main():
    fpath = os.path.join(os.pardir, os.pardir, 'data', 'data')
    combine_data(fpath)


if __name__ == '__main__':
    main()    

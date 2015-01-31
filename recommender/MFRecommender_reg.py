import numpy as np
import pandas as pd
import graphlab as gl
import os


def train(fpath):
    df = pd.read_csv(fpath)
    df = df.drop(['DateTime'], axis=1)
    df.SubId = np.object_(np.int64(df.SubId))
    df.UserId = np.object_(df.UserId)
    df.Rating = np.int64(df.Rating)
    # remove users with less than 50 ratings
    temp = df.UserId.value_counts()[df.UserId.value_counts() < 50].index
    temp = set(temp)
    remain = []
    for i in df.index:
        if df.UserId[i] not in temp:
            remain.append(i)
    df = df.loc[remain]
    # remove items with less than 50 ratings
    temp = df.SubId.value_counts()[df.SubId.value_counts() < 50].index
    temp = set(temp)
    remain = []
    for i in df.index:
        if df.SubId[i] not in temp:
            remain.append(i)
    df = df.loc[remain]

    sf = gl.SFrame(df)
    print 'finished reading in data'
    dataset, test = gl.recommender.util.random_split_by_user(sf,
                                                             user_id='UserId',
                                                             item_id='SubId',
                                                             item_test_proportion=0.2,
                                                             random_seed=2345)
    training, validate = gl.recommender.util.random_split_by_user(dataset,
                                                                  user_id='UserId',
                                                                  item_id='SubId',
                                                                  item_test_proportion=0.25,
                                                                  random_seed=3456)
    numf = [2 ** e for e in range(3, 8)]
    regl = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    res = {}
    min_rmse = 99999.0
    coor_min_rmse = (numf[0], regl[0])
    for j in numf:
        for i in regl:
            rcmder = gl.recommender.factorization_recommender.create(training,
                                                                     user_id='UserId',
                                                                     item_id='SubId',
                                                                     target='Rating',
                                                                     regularization=i,
                                                                     num_factors=j)
            res[(j, i)] = rcmder.evaluate(validate, metric='rmse',
                                          target='Rating')['rmse_overall']
            if res[(j, i)] < min_rmse:
                min_rmse = res[(j, i)]
                coor_min_rmse = (j, i)
    print res
    print 'best combination is {} with RMSE {}'.format(coor_min_rmse, min_rmse)
    rcmder = gl.recommender.factorization_recommender.create(dataset,
                                                             user_id='UserId',
                                                             item_id='SubId',
                                                             target='Rating',
                                                             regularization=coor_min_rmse[1],
                                                             num_factors=coor_min_rmse[0])
    print 'finished training model'
    print rcmder.evaluate(test, metric='rmse', target='Rating')
    return rcmder


def main():
    fpath = os.path.join(os.pardir, os.pardir, 'data', 'ratings.csv')
    rcmder = train(fpath)
    rcmder.save('model7')
    test(rcmder)


def test(rcmder):
    print rcmder.recommend(['glennq'], k=30)


if __name__ == '__main__':
    main()

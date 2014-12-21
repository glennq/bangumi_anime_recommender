import numpy as np
import pandas as pd
import graphlab as gl
import os
import matplotlib.pyplot as plt


def train(fpath):
    df = pd.read_csv(fpath)
    df = df.drop(['DateTime'], axis=1)
    df.SubId = np.object_(np.int64(df.SubId))
    df.UserId = np.object_(df.UserId)
    df = df[df.Rating != 'email__']
    df.Rating = np.int64(df.Rating)
    temp = df.UserId.value_counts()[df.UserId.value_counts() < 10].index
    temp = set(temp)
    remain = []
    for i in df.index:
        if df.UserId[i] not in temp:
            remain.append(i)
    df = df.loc[remain]
    sf = gl.SFrame(df)
    print 'finished reading in data'
    dataset, test = gl.recommender.util.random_split_by_user(sf,
                                                             user_id='UserId',
                                                             item_id='SubId',
                                                             target='Rating')
    training, validate = gl.recommender.util.random_split_by_user(dataset,
                                                                  user_id='UserId',
                                                                  item_id='SubId',
                                                                  target='Rating')
    regl = [10 ** e for e in range(-5, 6)]
    res = []
    for i in regl:
        rcmder = gl.recommender.factorization_recommender.create(training,
                                                                 user_id='UserId',
                                                                 item_id='SubId',
                                                                 target='Rating',
                                                                 regularization=i)
        res.append(rcmder.evaluate_rmse(validate))
    plt.plot(regl, res)
    min_regl = regl[res.indexOf(min(res))]
    rcmder = gl.recommender.factorization_recommender.create(dataset,
                                                             user_id='UserId',
                                                             item_id='SubId',
                                                             target='Rating',
                                                             regularization=min_regl)
    print 'finished training model'
    print rcmder.evaluate_rmse(test)
    return rcmder


def main():
    fpath = os.path.join(os.pardir, os.pardir, 'data', 'ratings_all.csv')
    rcmder = train(fpath)
    rcmder.save('model')
    test(rcmder)


def test(rcmder):
    print rcmder.recommend(['glennq'], k=30)


if __name__ == '__main__':
    main()

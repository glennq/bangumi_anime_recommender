import numpy as np
import pandas as pd
import graphlab as gl
import os


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
    training, test = gl.recommender.util.random_split_by_user(sf,
                                                              user_id='UserId',
                                                              item_id='SubId',
                                                              target='Rating')
    rcmder = gl.recommender.factorization_recommender.create(training,
                                                             user_id='UserId',
                                                             item_id='SubId',
                                                             target='Rating')
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

import numpy as np
from sklearn.datasets import fetch_mldata
import pandas as pd
import os


class utility_func:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        temp = utility_func.sigmoid(z)
        return temp * (1 - temp)

    @staticmethod
    def cross_entropy_cost(a, y):
        return np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a)) / len(a)

    @staticmethod
    def cross_entropy_sigmoid_delta(z, a, y):
        return utility_func.sigmoid_prime(z) * (a - y)

    @staticmethod
    def cross_entropy_sigmoid_delta_out(a, y):
        return a - y

    def __init__(self, activation, loss):
        if activation == 'sigmoid' and loss == 'cross entropy':
            self.cost = utility_func.cross_entropy_cost
            self.delta = utility_func.cross_entropy_sigmoid_delta
            self.delta_out = utility_func.cross_entropy_sigmoid_delta_out


class neural_network:
    def __init__(self, data, user='user', item='item', target='rating',
                 layers_sizes=[100, 10], activation='sigmoid', num_factors=20,
                 loss='cross entropy', regularization='l2', lmbda=1.0):
        """
        data must be a pandas dataframe
        """
        self.func = utility_func(activation, loss)
        self.layers_sizes = [2 * num_factors] + layers_sizes
        self.users = data[user].unique()
        self.items = data[item].unique()
        user_dict = {s: i for i, s in enumerate(self.users)}
        item_dict = {s: i for i, s in enumerate(self.items)}
        self.data_user = np.empty(len(data))
        self.data_item = np.empty(len(data))
        self.data_y = np.zeros((len(data), layers_sizes[-1]))
        for i in xrange(len(data)):
            self.data_user[i] = user_dict[data[user][i]]
            self.data_item[i] = item_dict[data[item][i]]
            self.data_y[i, int(data[target][i] - 1)] = 1.0
        self.initialize_param(num_factors)
        self.regl_term = (lambda w: lmbda * w) if regularization == 'l2' \
                         else (lambda w: lmbda * np.sign(w))
        self.target = target

    def initialize_param(self, num_factors):
        # initialize parameters for model
        self.u = np.random.randn(len(self.users), num_factors) / 2
        self.v = np.random.randn(len(self.items), num_factors) / 2

        # initialize weights for neural net
        self.weights = []
        self.biases = []
        for i in range(len(self.layers_sizes) - 1):
            self.biases.append(np.random.randn(self.layers_sizes[i+1]) / 2)
            self.weights.append(np.random.randn(self.layers_sizes[i+1],
                                                self.layers_sizes[i]) /
                                np.sqrt(self.layers_sizes[i]) / 2)

    def train(self, momentum=0.9, mom_incr=20, test_data=None,
              num_epoch=50, lrate=0.5, batch_size=200):
        """
        lrate stands for learning rate.
        momentum stands for stable momentum coefficient.
        mon_incr is the number of iterations after which mom_coef is
        increased to stable momentum coefficient
        """
        accuracy = [1.0 / self.layers_sizes[-1]]
        for i in range(num_epoch):
            mom_b = [np.zeros(e.shape) for e in self.biases]
            mom_w = [np.zeros(e.shape) for e in self.weights]
            mom_coef = 0.5
            for j in range(0, len(self.data_y) - batch_size, batch_size):
                grad_w, grad_b = self.get_mini_batch_grad(self.u[self.data_user[j:j+batch_size]],
                                                          self.v[self.data_item[j:j+batch_size]], 
                                                          self.data_y[j:j+batch_size])
                if j == mom_incr:
                    mom_coef = momentum
                for l in range(len(self.biases)):
                    mom_b[l] = mom_coef * mom_b[l] + lrate * grad_b[l]
                    mom_w[l] = mom_coef * mom_w[l] + \
                               lrate * (grad_w[l] + self.regl_term(
                                   self.weights[l]) / len(self.data))
                    self.biases[l] -= mom_b[l]
                    self.weights[l] -= mom_w[l]
            print 'finished epoch {}'.format(i)
            if test_data is None:
                accuracy.append(self.test_accuracy(self.u[self.data_user[j+batch_size:j+2*batch_size]],
                                                   self.v[self.data_item[j+batch_size:j+2*batch_size]], 
                                                   self.data_y[j+batch_size:j+2*batch_size]))
            # decrease learning rate when accuracy does not increase
            if accuracy[-1] < accuracy[-2] * 0.95:
                lrate *= 0.5
        return accuracy[1:]

    def get_mini_batch_grad(self, X1, X2, y):
        X = np.hstack(X1, X2)
        # feed forward
        zs = []
        a = [X]
        for j in range(len(self.biases)):
            zs.append(a[-1].dot(self.weights[j].transpose()) + self.biases[j])
            a.append(utility_func.sigmoid(zs[-1]))
        # back prop
        delta = [self.func.delta_out(a[-1], y)]
        for i in range(1, len(self.biases)):
            delta.append(delta[-1].dot(self.weights[-i]) *
                         utility_func.sigmoid_prime(zs[-i-1]))
        delta = delta[::-1]    # reverse delta
        grad_b = [e.mean(axis=0) for e in delta]
        grad_w = []
        for i in range(len(delta)):
            grad = np.dot(delta[i].transpose(), a[i]) / len(y)
            grad_w.append(grad)
        # return value
        return grad_w, grad_b

    def predict_proba(self, X):
        z = None
        a = X
        for j in range(len(self.biases)):
            z = a.dot(self.weights[j].transpose()) + self.biases[j]
            a = utility_func.sigmoid(z)
        return a

    def cal_cost(self, X, y):
        return self.func.cost(self.predict_proba(X), y)

    def predict(self, X):
        p = self.predict_proba(X)
        res = np.zeros(p.shape)
        for i in range(len(p)):
            res[i, np.argmax(p[i])] = 1.0
        return res

    def test_accuracy(self, X1, X2, y):
        X = np.hstack(X1, X2)
        yhat = self.predict(X)
        return np.sum(yhat * y) / len(y)


def main():
    # read MNIST data
    data_dir = os.path.join(os.pardir, 'data2')
    mnist = fetch_mldata('MNIST original', data_home=data_dir)
    index = range(len(mnist.data))
    np.random.shuffle(index)
    train_X = mnist.data[index[:60000]]
    test_X = mnist.data[index[60000:70000]]
    # transform target to vector form
    y = np.zeros((len(mnist.target), 10))
    for i in range(len(mnist.target)):
        y[i, int(mnist.target[i])] = 1.0
    train_y = y[index[:60000]]
    test_y = y[index[60000:70000]]
    # train neural net
    neural_net = neural_network([784, 100, 10])
    neural_net.train(train_X, train_y)
    print 'test accuracy = {}'.format(neural_net.test_accuracy(test_X, test_y))


if __name__ == '__main__':
    main()

import random

import numpy as np
from numpy.linalg import norm
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def compute_accuracy(preds, gt):

    tp = (preds == gt).sum()
    return tp / len(preds)


def clean_data(data, labels):
    N, D = data.shape

    # setup classifier
    cls = SGDClassifier(dim=D)
    cls.set_params(n_epochs=150, batch_size=100, lr=0.2, dropout=False)

    # fit data to model
    cls.fit(data, labels, data, labels)

    preds, p = cls.predict(data)
    idx = np.where(preds != labels)[0]

    p_wrong = p[idx]
    print(f"idx: {idx}, prob: {p_wrong}")

    # find the index where model prediction is wrong, but it has strong confidence.
    _id = np.where((preds != labels) & ((p > 0.8) | (p < 0.2)))[0]

    # Remove those elements
    new_data = np.delete(data, _id, axis=0)
    new_labels = np.delete(labels, _id)

    return new_data, new_labels


class SGDClassifier:

    def __init__(self, dim=30, n_cls=2, use_bias=True):

        # weight of the model
        self.w = np.zeros(dim)
        # bia param
        self.b = 0.0
        # number of classes
        self.n_cls = n_cls
        self.use_bias = use_bias

    def set_params(self, batch_size=32, lr=0.01, n_epochs=200, shuffle=True,
                   weight_penalty=True, alpha=0.001, dropout=False, dropout_p=0.5):

        # number of epochs to train
        self.n_epochs = n_epochs

        # whether to shuffle data after each epoch
        self.shuffle = shuffle

        # training hyper parameters
        self.batch_size = batch_size
        self.lr = lr

        # whether to use feature dropout
        self.dropout = dropout

        # the probability for dropout
        self.dropout_p = dropout_p

        # whether to add weight regularization
        self.weight_penalty = weight_penalty

        # weight penalty term if regularization is used, aka, weight decay
        self.alpha = alpha

    def fit(self, xtrain, ytrain, xval, yval):
        # train the model for n_epochs
        for i in range(self.n_epochs):
            train_loss = 0
            n_iter = 0
            for x_batch, y_batch in self.__get_batch(xtrain, ytrain):

                n_iter += 1
                train_loss_batch = self.__compute_loss(x_batch, y_batch)
                train_loss += train_loss_batch

                # compute and update gradient
                self.__compute_gradient(x_batch, y_batch)

            val_loss = self.__compute_loss(xval, yval)
            val_acc = compute_accuracy(self.predict(xval)[0], yval)

            print(f"epoch: {i}, train loss: {train_loss/n_iter}, val loss: {val_loss}, " +
                  f"val acc: {val_acc}")


    def __get_batch(self, x, y):
        n_samples = x.shape[0]
        x, y = x.copy(), y.copy()

        idx = list(range(n_samples))
        # shuffle the data
        if self.shuffle:
            random.shuffle(idx)

        # get a batch of data
        for i in range(0, n_samples, self.batch_size):
            batch_idx = idx[i:i+self.batch_size]

            xbatch, ybatch = x[batch_idx], y[batch_idx]
            if self.dropout:
                xbatch = self.__dropout(xbatch)

            yield xbatch, ybatch

    def __dropout(self, x):
        """
        dropout for a batch of training samples
        """
        # note that dropout is for each sample in the batch.
        p = np.random.random(x.shape)
        self.dropout_mask = np.where(p <= self.dropout_p, 0, 1)

        # element-wise multiplication
        x_new = x * self.dropout_mask / (1 - self.dropout_p)
        return x_new

    def predict(self, x):
        # probability output
        prob = 1 / (np.exp(-1* (np.dot(self.w, x.T) + self.b)) + 1)

        y_pred = np.where(prob <= 0.5, 0, 1)
        return y_pred, prob


    def __compute_loss(self, x, y):
        prob = 1 / (np.exp(-1* (np.dot(self.w, x.T) + self.b)) + 1)

        # to deal with nan for np.log()
        epsilon = 1e-8
        prob[np.isclose(prob, 0.0)] = epsilon
        prob[np.isclose(prob, 1.0)] = 1-epsilon

        # print(f"prob: {prob}")
        loss = -np.mean(y * np.log(prob) + (1-y) * np.log(1-prob)) + 0.5 * self.alpha * norm(self.w)

        return loss

    def __compute_gradient(self, x, y):
        # compute gradient over the batch
        prob = 1 / (np.exp(-1* (np.dot(self.w, x.T) + self.b)) + 1)

        n = x.shape[0]
        delta_w = 1.0 / n * np.dot((prob - y).T, x) + self.alpha * self.w
        delta_b = np.mean(prob - y)

        # update parameters
        self.w -= self.lr * delta_w
        self.b -= self.lr * delta_b


def data_aug(x, y):
    """
    Augment the data
    """
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    print(f"cls 0: {len(idx0)}, cls 1: {len(idx1)}")

    diff = len(idx1) - len(idx0)
    _idx = random.sample(list(idx0), k=diff)

    x_sampled = x[_idx]
    y_sampled = np.zeros(diff, dtype=np.int64)

    x_new = np.concatenate([x, x_sampled], axis=0)
    y_new = np.concatenate([y, y_sampled])

    # shuffle the data to make it random ordered
    n = x_new.shape[0]
    idx = list(range(n))
    random.shuffle(idx)
    x_new = x_new[idx]
    y_new = y_new[idx]

    return x_new, y_new


def run_training():

    # load dataset from scikit-learn
    data, labels = load_breast_cancer(return_X_y=True)
    N, D = data.shape

    print(f"N={N}, D={D}")

    # normalize the features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # clean the data
    data, labels = clean_data(data, labels)
    print(f"sample num: {data.shape[0]}")

    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=2022)

    x_train, y_train = data_aug(x_train, y_train)

    # setup classifier
    cls = SGDClassifier(dim=D)
    cls.set_params(n_epochs=150, batch_size=16, lr=0.1, dropout=True)

    # fit data to model
    cls.fit(data, labels, data, labels)

    preds = cls.predict(data)

    acc = compute_accuracy(preds, labels)

    print(f"Accuracy of model {acc*100:.2f}")


if __name__ == '__main__':

    run_training()

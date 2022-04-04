import numpy as np
import random
from sklearn.datasets import load_breast_cancer


def compute_accuracy(preds, gt):

    tp = (preds == gt).sum()
    return tp / len(preds)


def clean_data(data, labels):

    # apply simple heuristics on data, labels to do cleanup/validation if needed

    return data, labels


class SGDClassifier:

    def __init__(self, dim=30, n_cls=2, use_bias=True):

        # weight of the model
        self.w = np.zeros(dim)
        # bia param
        self.b = 0.0
        # number of classes
        self.n_cls = n_cls
        self.use_bias = use_bias

    def set_params(self, batch_size=32, lr=0.01, dropout=True, n_epocs=10, n_iters=1000):

        # training hyper parameters
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        # number of epocs, number of iterations per epoc
        self.n_epocs = n_epocs
        self.n_iter = n_iters

    def fit(self, x, y):

        pass

    def predict(self, x):

        return random.choices([0, 1], k=x.shape[0])

    def __compute_loss(self, x, y):

        # compute log-loss

        return 0.0

    def __compute_gradient(self, loss, x, y):

        # compute gradient over the batch

        return 0.0


def run_training():

    # load dataset from scikit-learn
    data, labels = load_breast_cancer(return_X_y=True)
    N, D = data.shape

    print(f"N={N}, D={D}")

    # setup classifier
    cls = SGDClassifier(dim=D)
    cls.set_params(batch_size=16, lr=0.001)

    # fit data to model
    cls.fit(data, labels)
    preds = cls.predict(data)

    acc = compute_accuracy(preds, labels)

    print(f"Accuracy of model {acc*100:.2f}")


if __name__ == '__main__':

    run_training()

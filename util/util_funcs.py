import numpy as np


def nhot(X, D = 10):
    assert isinstance(X, np.ndarray), "Expected np.ndarray, however, we got %s"%type(X)
    assert len(X.shape) == 2
    # TODO remove this ugly for-loop for something more efficient
    N,d = X.shape
    Y = np.zeros((N,D))
    for n in range(N):
        Y[n,X[n]] = 1.0/d
    return Y


def onehot(X, D = 10):
    assert isinstance(X, np.ndarray), "Expected np.ndarray, however, we got %s"%type(X)
    #TODO remove squeeze??
    X = np.squeeze(X)
    assert len(X.shape) == 1
    # TODO remove this ugly for-loop for something more efficient
    N = X.shape[0]
    Y = np.zeros((N,D))
    Y[range(N),X] = 1.0
    return Y

def acc_intersection(X,Y):
    """
    Calculate the accuracy.
    For the intersection, defined as the |intersection|/max(|X|,|Y|)
    calculated per row
    :param X:
    :param Y:
    :return:
    """
    N,D1 = X.shape
    N,D2 = Y.shape
    D = np.max((D1,D2))

    Z = np.hstack((X,Y))
    Z.sort(axis=1)
    acc = np.sum(Z[:,1:] == Z[:,:-1],axis=1)/float(D)
    return np.mean(acc)


class EWMA(object):
    def __init__(self,alpha):
        self.alpha = alpha
        self.value = None
        self.value_recent = None

    def add(self,x):
        if self.value is None:
            self.value = x
            self.value_recent = self.value.copy()
        else:
            self.value = self.alpha*self.value + (1-self.alpha)*x
            self.value_recent = x.copy()
        return

    @property
    def tup(self):
        return self.value_recent,self.value

    @property
    def val(self):
        return self.value

    def __str__(self):
        return "%ewma: 5.3f most recent value %5.3f"%(self.tup)

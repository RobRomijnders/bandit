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
        return self.value,self.value_recent

    @property
    def val(self):
        return self.value

    def __str__(self):
        return "%ewma: 5.3f most recent value %5.3f"%(self.value,self.value_recent)

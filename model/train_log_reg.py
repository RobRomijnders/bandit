
import matplotlib
matplotlib.use('TkAgg')
from util import input_data

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data")
input_data.MNISTDataSet.width = 14 #preferably a multiple of 4
input_data.MNISTDataSet.height = 14
input_data.MNISTDataSet.num_a = 1 #how many are active at some moment

batchsize = input_data.MNISTDataSet.bsz = 16

# X_train, y_train, r_train = mnist.train.random_policy(5)
# for i in range(5):
#     plt.imshow(X_train[i])
#     plt.show()
# print(y_train)
# print(r_train)

BSZ = 10000
X_train, y_train, r_train = mnist.train.random_policy(BSZ)
X_train = np.reshape(X_train,(BSZ,-1))
X_train = np.concatenate((X_train, y_train),1)

BSZ = 4500
X_val, y_val, r_val = mnist.val.random_policy(BSZ)
X_val = np.reshape(X_val,(BSZ,-1))
X_val = np.concatenate((X_val, y_val),1)

SVM = SVC(C=1.0, kernel='linear', gamma='auto', verbose=False)

SVM.fit(X_train, r_train)

print(SVM.score(X_val, r_val))



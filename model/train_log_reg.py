
import matplotlib
matplotlib.use('TkAgg')
from util import input_data

import numpy as np

mnist = input_data.read_data_sets("MNIST_data")
input_data.MNISTDataSet.width = 48 #preferably a multiple of 4
input_data.MNISTDataSet.height = 16
input_data.MNISTDataSet.num_a = 3 #how many are active at some moment

batchsize = input_data.MNISTDataSet.bsz = 16

im, lbl, _ = mnist.train.next_mix_batch(True)

X_train, y_train = mnist.train.next_mix_batch(make_hot=False)

import matplotlib
matplotlib.use('TkAgg')
from util import input_data

mnist = input_data.read_data_sets("MNIST_data")

images, labels = mnist.train.next_batch(16)

"""Plot some normal image"""
mnist.train.plot_example()

print('Get the batch')
im, lbl = mnist.train.next_mix_batch(16)
mnist.train.plot_example(True)


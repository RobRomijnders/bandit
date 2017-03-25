
import matplotlib
matplotlib.use('TkAgg')
from util import input_data
from model.conv import Model
from util.util_funcs import EWMA

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf  #version alpha 1.0
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data")

images, labels = mnist.train.next_batch(True)

"""Plot some normal image"""
# mnist.train.plot_example()

print('Get the batch')

cf = {}
cf['width'] = WIDTH = input_data.MNISTDataSet.width = 48 #preferably a multiple of 4
cf['num_a'] = NUM_A = input_data.MNISTDataSet.num_a = 3 #how many are active at some moment
cf['num_actions'] = 10 #how many possible actions
cf['lr'] = 0.005 #Static learning rate

dropout = 0.9
max_iterations = 1000
log_interval = 100
batchsize = input_data.MNISTDataSet.bsz = 16

im, lbl = mnist.train.next_mix_batch(True)
# mnist.train.plot_example(True)

model = Model(cf)

"""Training time"""
sess = tf.Session()
sess.run(model.init_op)

track_val = EWMA(0.8)
track_train = EWMA(0.9)

try:
    for i in range(max_iterations):
        X_batch, y_batch = mnist.train.next_mix_batch(True)
        cost,_,debug = sess.run([model.cost_batch,model.train_step,model.pred],{model.X:X_batch, model.y: y_batch, model.keep_prob:dropout})
        track_train.add(cost)
        print(y_batch)
        print(debug)

        if i%log_interval == 0:
            X_batch, y_batch = mnist.val.next_mix_batch(True)
            cost = sess.run(model.cost_batch,{model.X:X_batch, model.y:y_batch, model.keep_prob: 1.0})
            track_val.add(cost)
            print("At %7i/%7i for train %s for val %s"%(i,max_iterations,track_train.tup,track_val.tup))

except KeyboardInterrupt:
    print('Finished training')
finally:
    sess.close()


a=0

import matplotlib
matplotlib.use('TkAgg')
from util import input_data
from model.conv import Model
from util.util_funcs import EWMA, acc_intersection

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf  #version alpha 1.0
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data")

"""Plot some normal image"""
cf = {}
cf['width'] = WIDTH = input_data.MNISTDataSet.width = 48 #preferably a multiple of 4
cf['height'] = HEIGHT = input_data.MNISTDataSet.height = 16
cf['num_a'] = NUM_A = input_data.MNISTDataSet.num_a = 3 #how many are active at some moment
cf['num_actions'] = 10 #how many possible actions
cf['lr'] = 0.0005 #Static learning rate

dropout = 0.8
max_iterations = 10000
log_interval = 200
batchsize = input_data.MNISTDataSet.bsz = 16

im, lbl, _ = mnist.train.next_mix_batch(True)
# mnist.train.plot_example(True)

model = Model(cf)

"""Training time"""
sess = tf.Session()
sess.run(model.init_op)

track_val = EWMA(0.7)
track_train = EWMA(0.95)
track_acc = EWMA(0.7)

try:
    print('Start training')
    print('--Numbers for comparison:')
    print('  -default strategy yields entropy %5.3f'%(-1.0*np.log(1.0/10)))
    print('  -optimal strategy yields entropy %5.3f' % (-1.0 * np.log(1.0 / NUM_A)))
    for i in range(max_iterations):
        X_batch, y_batch, _ = mnist.train.next_mix_batch(True)
        cost,_,debug = sess.run([model.cost_batch,model.train_step,model.pred],{model.X:X_batch, model.y: y_batch, model.keep_prob:dropout})
        track_train.add(cost)

        if i%log_interval == 0:
            X_batch, y_batch, y_batch_ind = mnist.val.next_mix_batch(True)
            cost,pred = sess.run([model.cost_batch,model.pred],{model.X:X_batch, model.y:y_batch, model.keep_prob: 1.0})
            track_val.add(cost)
            acc = acc_intersection(pred, y_batch_ind)
            track_acc.add(acc)
            print("At %7i/%7i for train %5.3f/%5.3f for val %5.3f/%5.3f for acc %5.3f/%5.3f"%(i,max_iterations,*track_train.tup,*track_val.tup, *track_acc.tup))

except KeyboardInterrupt:
    print('Finished training')
finally:
    #Always make sure to close the tensorflow session
    sess.close()



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
cf['width'] = input_data.MNISTDataSet.width = 14
cf['height'] =input_data.MNISTDataSet.height =  14
cf['num_a'] = 1
cf['num_actions'] = 10 #how many possible actions
cf['lr'] = 0.001 #Static learning rate

NUM_A = 1

dropout = 0.6
max_iterations = 100000
log_interval = 200
batchsize = input_data.MNISTDataSet.bsz = 54

X, y, r = mnist.train.random_policy()
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
        X_batch, y_batch, r_batch = mnist.train.random_policy()
        cost,_,debug = sess.run([model.MAE,model.train_step,model.pred],{model.X:X_batch, model.y: y_batch,model.reward:r_batch,model.is_train:True, model.keep_prob:dropout})
        track_train.add(cost)

        if i%log_interval == 0:
            X_batch, y_batch, r_batch = mnist.train.random_policy()
            cost,pred = sess.run([model.MAE,model.pred],{model.X:X_batch, model.y:y_batch,model.reward:r_batch,model.is_train:False, model.keep_prob: 1.0})
            acc = np.mean(np.logical_and(pred > 0.5, r_batch))
            # if i > 3000:
            #     for n in range(10):
            #         print(pred[n], r_batch[n])
            track_val.add(cost)
            track_acc.add(acc)
            print("At %7i/%7i for train %5.3f/%5.3f for val %5.3f/%5.3f and acc %5.3f/%5.3f "%(i,max_iterations,*track_train.tup,*track_val.tup, *track_acc.tup))

except KeyboardInterrupt:
    print('Finished training')
finally:
    #Always make sure to close the tensorflow session
    sess.close()


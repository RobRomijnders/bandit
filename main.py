import matplotlib
matplotlib.use('TkAgg')
from util import dataloader
from model.conv import Model
from util.util_funcs import EWMA
import tensorflow as tf  #version 1.1.0rc1

mnist = dataloader.read_data_sets("MNIST_data",norm=True)
width, height = mnist.im_size

max_iterations = 10000
log_interval = 100
bsz = 34
dropout = 1.0
cf = {  'width'         :width,
        'height'        :height,
        'num_actions'   :10,
        'lr'            :0.001,
        'bsz'           :bsz,
        'lambda'        :0.0005,
        'POEM'          :True}

model = Model(cf)

#Mess up the policy
dataloader.pol0 = [0.15, 0.05, 0.15, 0.05, 0.15, 0.05, 0.15, 0.05, 0.15, 0.05]

"""Training time"""
sess = tf.Session()
sess.run(model.init_op)

track_val = EWMA(0.7)
track_train = EWMA(0.95)
# track_acc = EWMA(0.7)

try:
    print('Start training')
    print('Comparison: a default strategy yields -0.10')
    for i in range(max_iterations):
        X_batch, y_batch,r_batch,p_batch = mnist.next_lb_batch(bsz)
        cost,_,debug,db1,db2,db3 = sess.run([model.cost,
                                            model.train_step,
                                            model.pol_y,
                                            model.pi,
                                            model.pol_y,
                                            model.ind],
                                {model.X:           X_batch,
                                 model.y:           y_batch,
                                 model.reward:      r_batch,
                                 model.pol_0:       p_batch,
                                 model.is_train:    True,
                                 model.keep_prob:   dropout})
        track_train.add(cost)

        if i%log_interval == 0:
            X_batch, y_batch, r_batch, p_batch = mnist.next_lb_batch(bsz,'val')
            cost, debug = sess.run([model.cost,
                                    model.pol_y],
                                      {model.X: X_batch,
                                       model.y: y_batch,
                                       model.reward: r_batch,
                                       model.pol_0: p_batch,
                                       model.is_train: False,
                                       model.keep_prob: dropout})
            track_val.add(cost)
            print("At %7i/%7i for train %7.3f/%7.3f for val %7.3f/%7.3f"%(i,max_iterations,*track_train.tup,*track_val.tup))

except KeyboardInterrupt:
    print('Finished training')
finally:
    #Always make sure to close the tensorflow session
    sess.close()

a=0
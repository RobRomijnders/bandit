
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf  #version alpha 1.0
# tf.logging.set_verbosity(tf.logging.WARNING)
from tensorflow.python.ops import control_flow_ops


class Model():
    def __init__(self, cf):
        width = cf['width']
        height = cf['height']
        # num_a = cf['num_a']
        num_actions = cf['num_actions']
        lr = cf['lr']
        bsz = cf['bsz']
        channels = 1

        self.X = tf.placeholder(tf.float32, [bsz,height, width,channels],"input_image")
        self.y = tf.placeholder(tf.int32, [bsz],"action")
        self.keep_prob = tf.placeholder("float",name='keep_prob')
        self.is_train = tf.placeholder('bool')

        self.reward = tf.placeholder(tf.float32, [bsz])
        self.pol_0 = tf.placeholder(tf.float32, [bsz])

        n1 = 15
        n2 = 8

        n4 = 50
        with tf.variable_scope("conv", reuse=False) as scope:
            W1 = tf.get_variable('W1' ,[5, 5, 1, n1])
            b1 = tf.get_variable('b1' ,[n1,])
            a1 = tf.nn.conv2d(self.X, W1, [1, 2, 2, 1], padding='SAME') + b1
            u1 = tf.contrib.layers.batch_norm(a1,center=True,is_training=self.is_train)
            h1 = tf.nn.relu(tf.nn.dropout(u1,self.keep_prob))

            W3 = tf.get_variable('W3', [5, 5, n1, n2])
            b3 = tf.get_variable('b3', [n2, ])
            a3 = tf.nn.conv2d(h1, W3, [1, 1, 1, 1], padding='SAME') + b3
            a3d = tf.nn.dropout(a3, self.keep_prob)
            h3 = tf.nn.relu(tf.nn.dropout(a3d, self.keep_prob))

            L = int((width/2 )*int(height/2)*n2)
            W4 = tf.get_variable('W4' ,[L ,n4])
            b4 = tf.get_variable('b4' ,[n4 ,])
            h4_flat = tf.reshape(h3 ,[bsz ,L])
            u4_flat = tf.contrib.layers.batch_norm(h4_flat,center=True,is_training=self.is_train)
            a4 = tf.nn.xw_plus_b(u4_flat, W4, b4)  # in [batch_size, num_a]
            a4d = tf.nn.dropout(a4, self.keep_prob)
            h4 = tf.nn.relu(a4d)

            W5  = tf.get_variable('W5',[n4,num_actions])
            b5 = tf.get_variable('b5',[num_actions,])
            self.pi = tf.nn.softmax(tf.nn.xw_plus_b(h4,W5,b5))  #in [bsz,num_actions]


        with tf.variable_scope('CFRM', reuse=False) as scope:
            self.ind = tf.stack([tf.range(0,bsz),self.y],axis=1)
            self.pol_y = tf.gather_nd(self.pi,self.ind)                           #in [bsz,]
            importance = tf.divide(self.pol_y,self.pol_0)                    #in [bsz,]
            effective_sample_size = tf.stop_gradient(tf.reduce_sum(importance))                #scalar
            importance_sampling = tf.multiply(importance,self.reward)        #in [bsz,]
            self.SN_estimator = tf.reduce_sum(importance_sampling)/effective_sample_size #scalar

        with tf.variable_scope('opt', reuse=False) as scope:

            global_step = tf.Variable(0, trainable=False)
            lr_decay = tf.train.exponential_decay(lr, global_step, 5000000, 0.99, staircase=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_decay)

            self.cost = -1.0*self.SN_estimator
            self.train_step = optimizer.minimize(self.cost, global_step = global_step)

            self.init_op = tf.global_variables_initializer()
            self.all_saver = tf.train.Saver()

